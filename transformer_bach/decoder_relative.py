import os, sys
from datetime import datetime
from itertools import islice
import music21

import numpy as np
import torch
from transformer_bach.DatasetManager.helpers import PAD_SYMBOL, START_SYMBOL, END_SYMBOL, standard_name, standard_note
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import csv

from transformer_bach.utils import ensure_dir
from transformer_bach.small_bach_dataloader import SmallBachDataloaderGenerator
from transformer_bach.data_processor import DataProcessor
from transformer_bach.transformer_custom import TransformerEncoderLayerCustom, \
    TransformerDecoderLayerCustom, TransformerEncoderCustom, TransformerDecoderCustom, \
    TransformerCustom
from transformer_bach.utils import flatten, cuda_variable, categorical_crossentropy, \
    dict_pretty_print, to_numpy, top_k_top_p_filtering


class TransformerBach(nn.Module):
    def __init__(self,
                 model_dir,
                 train_dataloader_generator: SmallBachDataloaderGenerator,
                 val_dataloader_generator: SmallBachDataloaderGenerator,
                 data_processor: DataProcessor,
                 d_model,
                 num_encoder_layers,
                 num_decoder_layers,
                 n_head,
                 dim_feedforward,
                 positional_embedding_size,
                 num_channels,
                 num_events,
                 num_events_grouped,
                 dropout):
        """
        Like DecoderCustom, but the positioning is relative
        :param model_dir:
        :param dataloader_generator:
        :param data_processor:
        :param d_model:
        :param num_encoder_layers:
        :param num_decoder_layers:
        :param n_head:
        :param dim_feedforward:
        :param positional_embedding_size:
        :param num_channels:
        :param num_events:
        :param dropout:
        """
        super(TransformerBach, self).__init__()
        self.model_dir = model_dir

        self.train_dataloader_generator = train_dataloader_generator
        self.val_dataloader_generator = val_dataloader_generator
        self.data_processor = data_processor

        self.num_tokens_per_channel = self.data_processor.num_tokens_per_channel
        self.num_channels = len(self.num_tokens_per_channel)
        self.d_model = d_model
        # Compute num_tokens for source and target
        self.num_tokens_target = self.data_processor.num_tokens
        self.num_events_grouped = num_events_grouped
        assert num_events % self.num_events_grouped == 0

        self.channel_embeddings = nn.Parameter(
            torch.randn((1,
                         self.num_channels,
                         positional_embedding_size))
        )

        self.events_grouped_positioning_embeddings = nn.Parameter(
            torch.randn((1,
                         self.num_events_grouped,
                         positional_embedding_size))
        )

        #  Transformer
        encoder_layer = TransformerEncoderLayerCustom(
            d_model=d_model,
            nhead=n_head,
            attention_bias_type='relative_attention',
            num_channels=num_channels,
            num_events=num_events,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        decoder_layer = TransformerDecoderLayerCustom(
            d_model=d_model,
            nhead=n_head,
            attention_bias_type_self='relative_attention',
            attention_bias_type_cross='diagonal',
            num_channels_encoder=num_channels,
            num_events_encoder=num_events,
            num_channels_decoder=num_channels,
            num_events_decoder=num_events,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        custom_encoder = TransformerEncoderCustom(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        custom_decoder = TransformerDecoderCustom(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )

        self.transformer = TransformerCustom(
            d_model=self.d_model,
            nhead=n_head,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder
        )

        # Target embeddings is in data_processor
        self.linear_target = nn.Linear(self.data_processor.embedding_size
                                       + positional_embedding_size * 2,
                                       self.d_model)

        self.sos = nn.Parameter(torch.randn((1, 1, self.d_model)))

        self.mask_token = nn.Parameter(
            torch.randn((1, 1, self.data_processor.embedding_size))
        )

        self.pre_softmaxes = nn.ModuleList([nn.Linear(self.d_model, num_tokens_of_channel)
                                            for num_tokens_of_channel in self.num_tokens_per_channel
                                            ]
                                           )
        # optim
        self.optimizer = None

    def init_optimizers(self, lr=1e-3):
        self.optimizer = torch.optim.Adam(
            list(self.parameters())
            ,
            lr=lr
        )

    def __repr__(self):
        return 'DecoderRelative'

    def save(self, early_stopped, epoch=None):
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'
        
        if epoch:
            model_dir = f'{self.model_dir}/checkpoints/epoch-{epoch}'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(self.state_dict(), f'{model_dir}/decoder')
        # print(f'Model {self.__repr__()} saved')

    def load(self, early_stopped):
        print(f'Loading models {self.__repr__()}')
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'

        #  Deal with older configs
        if not os.path.exists(model_dir):
            model_dir = self.model_dir

        self.load_state_dict(torch.load(f'{model_dir}/decoder'))

    def add_positional_embedding(self, seq):
        batch_size, num_tokens, _ = seq.size()
        # add positional embeddings
        seq = torch.cat([
            seq,
            self.channel_embeddings.repeat(batch_size,
                                           num_tokens // self.num_channels,
                                           1),
            self.events_grouped_positioning_embeddings
                .repeat_interleave(self.num_channels, dim=1)
                .repeat((batch_size, num_tokens // (self.num_channels *
                                                    self.num_events_grouped), 1))
        ], dim=2)
        return seq

    def forward(self, x, masked_positions=None):
        """
        :param x: sequence of tokens (batch_size, num_events, num_channels)
        :param masked_positions: (batch_size, num_events, num_channels)
        None to get random masked_positions
        1 in masked positions when masked
        :return:
        """
        batch_size = x.size(0)

        # embed
        target = self.data_processor.preprocess(x)
        target_embedded = self.data_processor.embed(target)
        target_seq = flatten(target_embedded)

        # compute masked_x
        if masked_positions is not None:
            masked_positions = flatten(masked_positions)
        source_seq = self.mask(target_seq, masked_positions)

        # add positional embeddings and to d_model
        target_seq = self.add_positional_embedding(target_seq)
        target_seq = self.linear_target(target_seq)

        source_seq = self.add_positional_embedding(source_seq)
        source_seq = self.linear_target(source_seq)

        source_seq = source_seq.transpose(0, 1)
        target_seq = target_seq.transpose(0, 1)

        # shift target_seq by one
        dummy_input = self.sos.repeat(1, batch_size, 1)
        target_seq = torch.cat(
            [
                dummy_input,
                target_seq[:-1]
            ],
            dim=0)

        target_mask = cuda_variable(
            self._generate_square_subsequent_mask(target_seq.size(0))
        )
        memory_mask = target_mask + target_mask.t()
        source_mask = target_mask.t()

        output, attentions_decoder, attentions_encoder = self.transformer(
            source_seq,
            target_seq,
            src_mask=source_mask,
            tgt_mask=target_mask,
            memory_mask=memory_mask
        )

        output = output.transpose(0, 1).contiguous()

        output = output.view(batch_size,
                             -1,
                             self.num_channels,
                             self.d_model)
        weights_per_category = [
            pre_softmax(t[:, :, 0, :])
            for t, pre_softmax in zip(output.split(1, 2), self.pre_softmaxes)
        ]

        # we can change loss mask
        loss = categorical_crossentropy(
            value=weights_per_category,
            target=target,
            mask=torch.ones_like(target)
        )

        loss = loss.mean()
        return {
            'loss':                 loss,
            'attentions_decoder':   attentions_decoder,
            'attentions_encoder':   attentions_encoder,
            'weights_per_category': weights_per_category,
            'monitored_quantities': {
                'loss': loss.item()
            }
        }

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def mask(self, x, masked_indices=None):
        batch_size, num_tokens, embedding_dim = x.size()
        p = torch.rand(1).item() * 0.5
        if masked_indices is None:
            masked_indices = (torch.rand_like(x[:, :, 0]) > p).float()
        masked_indices = masked_indices.unsqueeze(2).repeat(1, 1,
                                                            embedding_dim)
        x = x * (1 - masked_indices) + self.mask_token * masked_indices
        return x

    def epoch(self, data_loader,
              train=True,
              num_batches=None,
              ):
        means = None

        if train:
            self.train()
        else:
            self.eval()
        
        for sample_id, tensor_dict in tqdm(enumerate(islice(data_loader, num_batches)), ncols=80):

            # ==========================
            with torch.no_grad():
                x = tensor_dict['x']

            # ========Train  =============
            self.optimizer.zero_grad()
            forward_pass = self.forward(
                x
            )
            loss = forward_pass['loss']
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                self.optimizer.step()

            # Monitored quantities
            monitored_quantities = forward_pass['monitored_quantities']
            
            # average quantities
            if means is None:
                means = {key: 0
                         for key in monitored_quantities}
            means = {
                key: value + means[key]
                for key, value in monitored_quantities.items()
            }

            del loss

        # renormalize monitored quantities
        means = {
            key: value / (sample_id + 1)
            for key, value in means.items()
        }
        return means

    def train_model(self,
                    batch_size,
                    num_batches=None,
                    num_epochs=10,
                    lr=1e-3,
                    plot=False,
                    num_workers=0,
                    **kwargs
                    ):
        if plot:
            self.writer = SummaryWriter(f'{self.model_dir}')

        self.init_optimizers(lr=lr)
        print(f'Train for {num_epochs} epochs')
        for epoch_id in range(num_epochs):
            generator_train = self.train_dataloader_generator.dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
            )

            generator_val = self.val_dataloader_generator.dataloaders(
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
            )

            monitored_quantities_train = self.epoch(
                data_loader=generator_train,
                train=True,
                num_batches=num_batches,
            )
            del generator_train

            monitored_quantities_val = self.epoch(
                data_loader=generator_val,
                train=False,
                num_batches=num_batches // 2 if num_batches is not None else None,
            )
            del generator_val

            train_loss = monitored_quantities_train['loss']
            val_loss = monitored_quantities_val['loss']

            print(f'======= Epoch {epoch_id} =======')
            print(f'---Train---')
            dict_pretty_print(monitored_quantities_train, endstr=' ' * 5)
            print()
            print(f'---Val---')
            dict_pretty_print(monitored_quantities_val, endstr=' ' * 5)
            print('\n')

            if plot:
                self.plot(epoch_id,
                          monitored_quantities_train,
                          monitored_quantities_val)
        
        return train_loss, val_loss

    def plot(self, epoch_id, monitored_quantities_train,
             monitored_quantities_val):
        for k, v in monitored_quantities_train.items():
            self.writer.add_scalar(f'{k}/train', v, epoch_id)
        for k, v in monitored_quantities_val.items():
            self.writer.add_scalar(f'{k}/val', v, epoch_id)

    def generate_fixed_size(self, temperature, batch_size=1, melody_constraint=None):
        self.eval()

        with torch.no_grad():
            # x = self.init_generation(num_events=self.data_processor.num_events)
            x, masked_positions = self.init_generation_chorale(
                num_events=self.data_processor.num_events,
                start_index=4,
                melody_constraint=melody_constraint)

            # Duplicate along batch dimension
            x = x.repeat(batch_size, 1, 1)
            masked_positions = masked_positions.repeat(batch_size, 1, 1)

            for event_index in range(self.data_processor.num_events):
                for channel_index in range(self.num_channels):

                    forward_pass = self.forward(x, masked_positions=masked_positions)

                    weights_per_voice = forward_pass['weights_per_category']
                    weights = weights_per_voice[channel_index]

                    logits = weights[:, event_index, :]
                    logits = torch.stack([top_k_top_p_filtering(logit,
                                                                top_p=0.9)
                                          for logit in logits
                                          ],
                                         dim=0)
                    logits = logits / temperature
                    p = torch.softmax(logits, dim=1)
                    p = to_numpy(p)
                    # update generated sequence
                    for batch_index in range(batch_size):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel[channel_index]
                        ), p=p[batch_index])

                        # copy melody?
                        # if masked_positions[batch_index, event_index, channel_index] == 1:
                        #     x[batch_index, event_index, channel_index] = int(new_pitch_index)

                        x[batch_index, event_index, channel_index] = int(new_pitch_index)

        # to score
        tensor_score = self.data_processor.postprocess(
            x.cpu().split(1, 0))
        scores = self.train_dataloader_generator.to_score(tensor_score)

        # save scores in model_dir
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(f'{self.model_dir}/generations'):
            os.mkdir(f'{self.model_dir}/generations')

        for k, score in enumerate(scores):
            score.write('xml', f'{self.model_dir}/generations/{timestamp}_{k}.xml')

        return scores

    def generate(self, 
                 temperature=1,
                 top_p=1,
                 batch_size=1,
                 melody_constraint=None,
                 hard_constraint=False):
        """
        temperature: 
        top_p: parameter for top p sampling
        batch_size: number of generations
        melody_constraint: soprano line
        hard_constraint: whether constraint is false
        """
        self.eval()
        if melody_constraint is not None:
            num_events = len(melody_constraint)
        else:
            num_events = self.data_processor.num_events * 4

        num_start_index = 4
        with torch.no_grad():
            num_events = num_events + 2 * num_start_index
            x, masked_positions = self.init_generation_chorale(
                num_events=num_events,
                start_index=num_start_index,
                melody_constraint=melody_constraint)
            # Duplicate along batch dimension
            x = x.repeat(batch_size, 1, 1)
            masked_positions = masked_positions.repeat(batch_size, 1, 1)
            for event_index in range(num_events):
                for channel_index in range(self.num_channels):
                    # slice x
                    t_begin, t_end, t_relative = self.compute_start_end_times(
                        event_index // self.num_events_grouped,
                        num_blocks=num_events // self.num_events_grouped,
                        num_blocks_model=self.num_tokens_target // (self.num_events_grouped
                                                                    * self.num_channels)
                    )

                    event_begin = t_begin * self.num_events_grouped
                    event_end = t_end * self.num_events_grouped
                    event_relative = (t_relative * self.num_events_grouped + event_index % self.num_events_grouped)

                    x_slice = x[:, event_begin: event_end]
                    masked_positions_slice = masked_positions[:, event_begin: event_end]
                    forward_pass = self.forward(x_slice, masked_positions=masked_positions_slice)

                    weights_per_voice = forward_pass['weights_per_category']
                    weights = weights_per_voice[channel_index]

                    logits = weights[:, event_relative, :]

                    # temperature before or after?
                    logits = logits / temperature
                    logits = torch.stack([top_k_top_p_filtering(logit,
                                                                top_p=top_p)
                                          for logit in logits
                                          ],
                                         dim=0)
                    p = torch.softmax(logits, dim=1)
                    p = to_numpy(p)

                    # update generated sequence
                    for batch_index in range(batch_size):
                        new_pitch_index = np.random.choice(np.arange(self.num_tokens_per_channel[channel_index]), 
                                                                     p=p[batch_index])

                        # only if not constrained
                        if not (hard_constraint and masked_positions[batch_index, event_index, channel_index] == 0):
                            x[batch_index, event_index, channel_index] = int(new_pitch_index)

        # to score
        x = x[:, num_start_index:-num_start_index, :]                               # remove padding
        tensor_score = self.data_processor.postprocess(x.cpu().split(1, 0))         # convert from tensor to numpy array
        scores = self.train_dataloader_generator.to_score(tensor_score)

        for score in scores:
            self.trim_generated_chorale(score)

        return scores

    def init_generation_chorale(self, num_events, start_index, melody_constraint=None):
        PAD = [d[PAD_SYMBOL] for d in self.train_dataloader_generator.dataset.note2index_dicts]
        START = [d[START_SYMBOL] for d in self.train_dataloader_generator.dataset.note2index_dicts]
        END = [d[END_SYMBOL] for d in self.train_dataloader_generator.dataset.note2index_dicts]
        aa = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(1, start_index - 1, 1).long()
        bb = torch.Tensor(START).unsqueeze(0).unsqueeze(0).long().repeat(1, num_events - 2 * start_index + 1, 1).long()
        cc = torch.Tensor(END).unsqueeze(0).unsqueeze(0).long()
        dd = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(1, start_index - 1, 1).long()
        init_sequence = torch.cat([aa, bb, cc, dd], 1)

        masked_positions = torch.ones_like(init_sequence)
        masked_positions[:, : start_index, :] = 0
        masked_positions[:, -start_index:, :] = 0

        if melody_constraint is not None:
            MELODY_CONSTRAINT = [self.train_dataloader_generator.dataset.note2index_dicts[0][note]
                                 for note in melody_constraint]
            for i in range(num_events - 2 * start_index):
                init_sequence[:, i + start_index, 0] = MELODY_CONSTRAINT[i]
                masked_positions[:, i + start_index, 0] = 0
        return cuda_variable(init_sequence), cuda_variable(masked_positions)

    def trim_generated_chorale(self, score):
        """
        :return: score with everything after and including the END token trimmed in place
        """
        for part in score.parts:
            found = music21.search.noteNameSearch(part, [standard_note(END_SYMBOL)])   # look for end note
            # assert len(found) in [0, 1]       # either there's exactly one end note, or none
            assert part.isSorted  
            if found:
                end_idx = found[0]
                num_pops = len(part) - end_idx      # total number of notes to 
                for _ in range(num_pops):
                    part.pop(end_idx)
                    
    def compute_start_end_times(self, t, num_blocks, num_blocks_model):
        """

        :param t:
        :param num_blocks: num_blocks of the sequence to be generated
        :param num_blocks_model:
        :return:
        """
        # t_relative
        if num_blocks_model // 2 <= t < num_blocks - num_blocks_model // 2:
            t_relative = (num_blocks_model // 2)
        else:
            if t < num_blocks_model // 2:
                t_relative = t
            elif t >= num_blocks - num_blocks_model // 2:
                t_relative = num_blocks_model - (num_blocks - t)
            else:
                NotImplementedError

        # choose proper block to use
        t_begin = min(max(0, t - num_blocks_model // 2), num_blocks - num_blocks_model)
        t_end = t_begin + num_blocks_model

        return t_begin, t_end, t_relative
