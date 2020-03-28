import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from torch.distributions import Normal, kl_divergence
import pickle


def cuda_variable(tensor):
    if torch.cuda.is_available():
        return tensor.to('cuda')
    else:
        return tensor


def to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()


def dict_pretty_print(d, endstr='\n'):
    for key, value in d.items():
        if type(value) == list:
            print(f'{key.capitalize()}: [%s]' % ', '.join(map(str, value)))
        else:
            print(f'{key.capitalize()}: {value:.6}', end=endstr)


def chorale_accuracy(value, target):
    """
    :param value: list of (batch_size, chorale_length, num_notes)
    :param target: (batch_size, num_voices, chorale_length)
    :return:
    """
    batch_size, num_voices, chorale_length = target.size()
    batch_size, chorale_length, _ = value[0].size()
    num_voices = len(value)

    # put num_voices first
    target = target.transpose(0, 1)

    sum = 0
    for voice, voice_target in zip(value, target):
        max_values, max_indexes = torch.max(voice, dim=2, keepdim=False)
        num_correct = (max_indexes == voice_target).float().mean().item()
        sum += num_correct

    return sum / num_voices


def categorical_crossentropy(value, target, mask=None):
    """

    :param value: list of (batch_size, num_events, num_tokens_of_corresponding_channel)
    :param target: (batch_size, num_events, num_channels)
    :param mask: (batch_size, num_events, num_channels)
    :return:
    """
    cross_entropy = nn.CrossEntropyLoss(size_average=False, reduce=False)
    sum = 0

    for channel_probs, target_channel, mask_channel in zip(value,
                                                           target.split(1, dim=2),
                                                           mask.split(1, dim=2)):
        # select relevent indices
        batch_size, num_events, num_tokens_of_channel = channel_probs.size()
        num_events_mask = mask_channel.sum() // batch_size
        assert mask_channel.sum() % batch_size == 0

        probs = channel_probs[mask_channel.bool().repeat(1, 1, num_tokens_of_channel)]
        target = target_channel[mask_channel.bool()]

        ce = cross_entropy(probs.view(-1, num_tokens_of_channel),
                           target.view(-1))
        sum = sum + ce
    return sum


def flatten(x):
    """

    :param x:(batch, num_events, num_channels, ...)
    :return: (batch, num_events * num_channels, ...) with num_channels varying faster
    """
    size = x.size()
    assert len(size) >= 3
    batch_size, num_events, num_channels = size[:3]
    remaining_dims = list(size[3:])
    x = x.view(batch_size, num_events * num_channels, *remaining_dims)
    return x


def unflatten(sequence, num_channels):
    """

    :param sequence: (batch_size, num_events * num_channels, ...)
    where num_channels is varying faster
    :return: (batch_size, num_events, num_channels, ...)
    """
    size = sequence.size()
    assert len(size) >= 2
    batch_size, sequence_length = size[:2]
    assert sequence_length % num_channels == 0
    num_events = sequence_length // num_channels
    remaining_dims = list(size[2:])

    chorale = sequence.view(batch_size, num_events, num_channels, *remaining_dims)
    return chorale


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def ensure_dir(directory):
    """
    create directory if it does not already exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_threshold(data_file=None, col=-1):
    thres = np.NINF         # minimum score seen so far
    
    with open(data_file, 'r') as fin:
        next(fin)
        for row in fin:
            s = float(row.split(',')[col])
            if s > thres:
                thres = s
    return thres