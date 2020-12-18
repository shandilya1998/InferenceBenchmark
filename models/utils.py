import torch
import math

from torch import nn
from torch.nn import functional as F

import logging.config
import os
import random
import sys
import time
from contextlib import contextmanager

import dllogger
import numpy as np
import torch.distributed as dist
import torch.nn.init as init
import torch.utils.collect_env

class BatchNorm2d(nn.Module):
    """
    Fixed version of BatchNorm2d, which has only the scale and bias
    """

    def __init__(self, out):
        super(BatchNorm2d, self).__init__()
        self.register_buffer("scale", torch.ones(out))
        self.register_buffer("bias", torch.zeros(out))

    #@torch.jit.script_method
    def forward(self, x):
        scale = self.scale.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return x * scale + bias


class BiasAdd(nn.Module):
    """
    Fixed version of BatchNorm2d, which has only the scale and bias
    """

    def __init__(self, out):
        super(BiasAdd, self).__init__()
        self.register_buffer("bias", torch.zeros(out))

    #@torch.jit.script_method
    def forward(self, x):
        bias = self.bias.view(1, -1, 1, 1)
        return x + bias


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        #import pdb; pdb.set_trace()
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


def box_area(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def box_iou(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = box_area(overlap_left_top, overlap_right_bottom)
    area0 = box_area(boxes0[..., :2], boxes0[..., 2:])
    area1 = box_area(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def nms(box_scores, iou_threshold):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = box_iou(rest_boxes, current_box.unsqueeze(0))
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


@torch.jit.script
def decode_boxes(rel_codes, boxes, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor

    # perform some unpacking to make it JIT-fusion friendly
    
    #rel_codes=rel_codes[0][None]
    wx = weights[1]
    wy = weights[0]
    ww = weights[3]
    wh = weights[2]

    boxes_x1 = boxes[:, 1].unsqueeze(1).unsqueeze(0)
    boxes_y1 = boxes[:, 0].unsqueeze(1).unsqueeze(0)
    boxes_x2 = boxes[:, 3].unsqueeze(1).unsqueeze(0)
    boxes_y2 = boxes[:, 2].unsqueeze(1).unsqueeze(0)

    dx = rel_codes[:,:, 1].unsqueeze(2)
    dy = rel_codes[:,:, 0].unsqueeze(2)
    dw = rel_codes[:,:, 3].unsqueeze(2)
    dh = rel_codes[:,:, 2].unsqueeze(2)

    # implementation starts here
    widths = boxes_x2 - boxes_x1
    heights = boxes_y2 - boxes_y1
    ctr_x = boxes_x1 + 0.5 * widths
    ctr_y = boxes_y1 + 0.5 * heights

    dx = dx / wx
    dy = dy / wy
    dw = dw / ww
    dh = dh / wh

    pred_ctr_x = dx * widths + ctr_x
    #import pdb; pdb.set_trace()
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.cat(
        [
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ],
        dim=2,
    )
    #import pdb; pdb.set_trace()
    return pred_boxes

def init_lstm_(lstm, init_weight=0.1):
    """
    Initializes weights of LSTM layer.
    Weights and biases are initialized with uniform(-init_weight, init_weight)
    distribution.
    :param lstm: instance of torch.nn.LSTM
    :param init_weight: range for the uniform initializer
    """
    # Initialize hidden-hidden weights
    init.uniform_(lstm.weight_hh_l0.data, -init_weight, init_weight)
    # Initialize input-hidden weights:
    init.uniform_(lstm.weight_ih_l0.data, -init_weight, init_weight)

    # Initialize bias. PyTorch LSTM has two biases, one for input-hidden GEMM
    # and the other for hidden-hidden GEMM. Here input-hidden bias is
    # initialized with uniform distribution and hidden-hidden bias is
    # initialized with zeros.
    init.uniform_(lstm.bias_ih_l0.data, -init_weight, init_weight)
    init.zeros_(lstm.bias_hh_l0.data)

    if lstm.bidirectional:
        init.uniform_(lstm.weight_hh_l0_reverse.data, -init_weight, init_weight)
        init.uniform_(lstm.weight_ih_l0_reverse.data, -init_weight, init_weight)

        init.uniform_(lstm.bias_ih_l0_reverse.data, -init_weight, init_weight)
        init.zeros_(lstm.bias_hh_l0_reverse.data)


def generate_seeds(rng, size):
    """
    Generate list of random seeds
    :param rng: random number generator
    :param size: length of the returned list
    """
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
    return seeds


def broadcast_seeds(seeds, device):
    """
    Broadcasts random seeds to all distributed workers.
    Returns list of random seeds (broadcasted from workers with rank 0).
    :param seeds: list of seeds (integers)
    :param device: torch.device
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        seeds_tensor = torch.tensor(seeds, dtype=torch.int64, device=device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seeds = seeds_tensor.tolist()
    return seeds


def setup_seeds(master_seed, epochs, device):
    """
    Generates seeds from one master_seed.
    Function returns (worker_seeds, shuffling_seeds), worker_seeds are later
    used to initialize per-worker random number generators (mostly for
    dropouts), shuffling_seeds are for RNGs resposible for reshuffling the
    dataset before each epoch.
    Seeds are generated on worker with rank 0 and broadcasted to all other
    workers.
    :param master_seed: master RNG seed used to initialize other generators
    :param epochs: number of epochs
    :param device: torch.device (used for distributed.broadcast)
    """
    if master_seed is None:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2**32 - 1)
        if get_rank() == 0:
            # master seed is reported only from rank=0 worker, it's to avoid
            # confusion, seeds from rank=0 are later broadcasted to other
            # workers
            logging.info(f'Using random master seed: {master_seed}')
    else:
        # master seed was specified from command line
        logging.info(f'Using master seed from command line: {master_seed}')

    # initialize seeding RNG
    seeding_rng = random.Random(master_seed)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(seeding_rng, get_world_size())

    # generate seeds for data shuffling, one seed for every epoch
    shuffling_seeds = generate_seeds(seeding_rng, epochs)

    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device)
    shuffling_seeds = broadcast_seeds(shuffling_seeds, device)
    return worker_seeds, shuffling_seeds


def barrier():
    """
    Call torch.distributed.barrier() if distritubed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank


def get_world_size():
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


@contextmanager
def sync_workers():
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    rank = get_rank()
    yield rank
    barrier()


@contextmanager
def timer(name, ndigits=2, sync_gpu=True):
    if sync_gpu:
        torch.cuda.synchronize()
    start = time.time()
    yield
    if sync_gpu:
        torch.cuda.synchronize()
    stop = time.time()
    elapsed = round(stop - start, ndigits)
    logging.info(f'TIMER {name} {elapsed}')


def setup_logging(log_all_ranks=True, log_file=os.devnull):
    """
    Configures logging.
    By default logs from all workers are printed to the console, entries are
    prefixed with "N: " where N is the rank of the worker. Logs printed to the
    console don't include timestaps.
    Full logs with timestamps are saved to the log_file file.
    """
    class RankFilter(logging.Filter):
        def __init__(self, rank, log_all_ranks):
            self.rank = rank
            self.log_all_ranks = log_all_ranks

        def filter(self, record):
            record.rank = self.rank
            if self.log_all_ranks:
                return True
            else:
                return (self.rank == 0)

    rank = get_rank()
    rank_filter = RankFilter(rank, log_all_ranks)

    logging_format = "%(asctime)s - %(levelname)s - %(rank)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG,
                        format=logging_format,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(rank)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('').addFilter(rank_filter)


def setup_dllogger(enabled=True, filename=os.devnull):
    rank = get_rank()

    if enabled and rank == 0:
        backends = [
            dllogger.JSONStreamBackend(
                dllogger.Verbosity.VERBOSE,
                filename,
                ),
            ]
        dllogger.init(backends)
    else:
        dllogger.init([])


def set_device(cuda, local_rank):
    """
    Sets device based on local_rank and returns instance of torch.device.
    :param cuda: if True: use cuda
    :param local_rank: local rank of the worker
    """
    if cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def init_distributed(cuda):
    """
    Initializes distributed backend.
    :param cuda: (bool) if True initializes nccl backend, if False initializes
        gloo backend
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = (world_size > 1)
    if distributed:
        backend = 'nccl' if cuda else 'gloo'
        dist.init_process_group(backend=backend,
                                init_method='env://')
        assert dist.is_initialized()
    return distributed


def log_env_info():
    """
    Prints information about execution environment.
    """
    logging.info('Collecting environment information...')
    env_info = torch.utils.collect_env.get_pretty_env_info()
    logging.info(f'{env_info}')


def pad_vocabulary(math):
    if math == 'tf32' or math == 'fp16' or math == 'manual_fp16':
        pad_vocab = 8
    elif math == 'fp32':
        pad_vocab = 1
    return pad_vocab


def benchmark(test_acc, target_acc, test_perf, target_perf):
    def test(achieved, target, name):
        passed = True
        if target is not None and achieved is not None:
            logging.info(f'{name} achieved: {achieved:.2f} '
                         f'target: {target:.2f}')
            if achieved >= target:
                logging.info(f'{name} test passed')
            else:
                logging.info(f'{name} test failed')
                passed = False
        return passed

    passed = True
    passed &= test(test_acc, target_acc, 'Accuracy')
    passed &= test(test_perf, target_perf, 'Performance')
    return passed


def debug_tensor(tensor, name):
    """
    Simple utility which helps with debugging.
    Takes a tensor and outputs: min, max, avg, std, number of NaNs, number of
    INFs.
    :param tensor: torch tensor
    :param name: name of the tensor (only for logging)
    """
    logging.info(name)
    tensor = tensor.detach().float().cpu().numpy()
    logging.info(f'MIN: {tensor.min()} MAX: {tensor.max()} '
                 f'AVG: {tensor.mean()} STD: {tensor.std()} '
                 f'NAN: {np.isnan(tensor).sum()} INF: {np.isinf(tensor).sum()}')


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, warmup=0, keep=False):
        self.reset()
        self.warmup = warmup
        self.keep = keep

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.iters = 0
        self.vals = []

    def update(self, val, n=1):
        self.iters += 1
        self.val = val

        if self.iters > self.warmup:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            if self.keep:
                self.vals.append(val)

    def reduce(self, op):
        """
        Reduces average value over all workers.
        :param op: 'sum' or 'mean', reduction operator
        """
        if op not in ('sum', 'mean'):
            raise NotImplementedError

        distributed = (get_world_size() > 1)
        if distributed:
            backend = dist.get_backend()
            cuda = (backend == dist.Backend.NCCL)

            if cuda:
                avg = torch.cuda.FloatTensor([self.avg])
                _sum = torch.cuda.FloatTensor([self.sum])
            else:
                avg = torch.FloatTensor([self.avg])
                _sum = torch.FloatTensor([self.sum])

            dist.all_reduce(avg)
            dist.all_reduce(_sum)
            self.avg = avg.item()
            self.sum = _sum.item()

            if op == 'mean':
                self.avg /= get_world_size()
                self.sum /= get_world_size()
