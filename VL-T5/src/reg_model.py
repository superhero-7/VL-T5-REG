import torch
import torch.nn as nn
import copy
# from memory_profiler import profile
from undecorated import undecorated
from types import MethodType
from torch.nn import NLLLoss, CrossEntropyLoss
import os.path as osp
import sys
if osp.join('/workspace/yfl/codebase/retr/pyutils', 'refer2', 'evaluation') not in sys.path:
    sys.path.insert(0, osp.join('/workspace/yfl/codebase/retr/pyutils', 'refer2', 'evaluation'))
from cider.cider import Cider
from tokenizer.ptbtokenizer import PTBTokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, TemperatureLogitsWarper


from modeling_t5 import VLT5

class VLT5REG(VLT5):
    def __init__(self, config):
        super().__init__(config)

    # @profile
    def train_step(self, batch, use_mmi=False, epoch=None, lama=1, margin=0.5):

        device = next(self.parameters()).device
        if use_mmi:
            vis_feats = torch.squeeze(batch['vis_feats'][:, 0].to(device))
            vis_pos = torch.squeeze(batch['boxes'][:, 0].to(device))

            neg_vis_feats = torch.squeeze(batch['vis_feats'][:, 1].to(device))
            neg_vis_pos = torch.squeeze(batch['boxes'][:, 1].to(device))

            input_ids = batch['input_ids'][:].to(device)

            lm_labels = batch["target_ids"].to(device)

            reduce_loss = True
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

            neg_output = self(
                input_ids=input_ids,
                vis_inputs=(neg_vis_feats, neg_vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

            lm_mask = lm_labels != -100
            B, L = lm_labels.size()

            pos_loss = output['loss']
            neg_loss = neg_output['loss']

            # 这里一会改还不知道能不能跑起来...
            if epoch % 10 == 0:
                margin /= 2
            loss = pos_loss + lama * (max(0, margin + pos_loss - neg_loss))

            result = {
                'loss': loss
            }
            return result
        else:
            vis_feats = batch['vis_feats'].to(device)
            input_ids = batch['input_ids'].to(device)
            vis_pos = batch['boxes'].to(device)

            lm_labels = batch["target_ids"].to(device)

            reduce_loss = True
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

            lm_mask = lm_labels != -100
            B, L = lm_labels.size()

            loss = output['loss']

            result = {
                'loss': loss
            }
            return result

    def rl_train_step(self, batch):

        reslut = {}
        criterion = CrossEntropyLoss(reduction='none', ignore_index=0)
        rewarder = Cider()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        target_sents = batch['target_texts']  # list:batch_size
        bs = len(target_sents)


        generate_with_grad = undecorated(self.generate)
        self.generate_with_grad = MethodType(generate_with_grad, self)

        output = self.generate_with_grad(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            max_length=20,
        )
        output_sents = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        # print('output_sents:', len(output_sents))
        # original scores: tuple: (tensor_matrix1, ..., tensor_matrixT) tensor_matrix_i: batch_size*vocab_size
        scores = torch.stack(output.scores, dim=0).permute(1, 0, 2)  # batch_size*sentence_len*vocabulary
        scores = scores.reshape(-1, scores.size(-1))
        target = output.sequences[:, 1:].reshape(-1)
        # index = target != 0
        # print(scores[list(range(len(scores))), target[index]])

        loss = criterion(scores,
                         target,
                         )
        loss = loss.view(bs, -1)
        loss = torch.mean(loss, dim=1)

        output_sents_dict = {}
        for ref_id, output_sent in zip(list(range(len(output_sents))), output_sents):
            output_sents_dict[str(ref_id)] = [output_sent]
        # print('output_sents_dict:', len(output_sents_dict))
        target_sents_dict = {}
        for ref_id, target_sent in zip(list(range(len(target_sents))), target_sents):
            target_sents_dict[str(ref_id)] = [target_sent]
        # print('target_sent_dict', len(target_sents_dict))
        # It seems change nothing bt PTBTokenizer,emmmmmm还是有点用的
        tokenizer = PTBTokenizer()
        output_sents_dict = tokenizer.tokenize(output_sents_dict)
        target_sents_dict = tokenizer.tokenize(target_sents_dict)
        # print('output_sents_dict after tokenize', len(output_sents_dict))
        # print('target_sents_dict after tokenize', len(target_sents_dict))

        output_reward, output_rewards = rewarder.compute_score(target_sents_dict, output_sents_dict)
        output_rewards = torch.from_numpy(output_rewards).to(device)
        # print('output_rewards:', len(output_rewards))

        # logits_warper = LogitsProcessorList([])

        greedy_output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            do_sample=False,
            max_length=20,
        )
        greedy_sents = self.tokenizer.batch_decode(greedy_output, skip_special_tokens=True)

        greedy_sents_dict = {}
        for idx, greedy_sent in zip(list(range(len(greedy_sents))), greedy_sents):
            greedy_sents_dict[str(idx)] = [greedy_sent]

        greedy_reward, greedy_rewards = rewarder.compute_score(target_sents_dict, greedy_sents_dict)
        reward_baseline = torch.from_numpy(greedy_rewards).to(device)
        # print(output_rewards.size(), reward_baseline.size())
        loss = (output_rewards-reward_baseline)*loss
        loss = loss.mean()
        reslut['loss'] = loss

        return reslut

    def rl_train_step2(self, batch):

        reslut = {}
        criterion = CrossEntropyLoss(reduction='none')
        rewarder = Cider()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        target_sents = batch['target_texts']  # list:batch_size


        generate_with_grad = undecorated(self.generate)
        self.generate_with_grad = MethodType(generate_with_grad, self)

        output = self.generate_with_grad(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            output_scores=True,
            return_dict_in_generate=True
        )
        output_sents = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        # print('output_sents:', len(output_sents))
        scores = torch.stack(output.scores, dim=0).permute(1, 0, 2)
        loss = criterion(scores.reshape(-1, scores.size(-1)),
                         output.sequences[:, 1:].reshape(-1),
                         )
        loss = loss.view(len(target_sents), -1)
        loss = torch.mean(loss, dim=1)

        output_sents_dict = {}
        for ref_id, output_sent in zip(list(range(len(output_sents))), output_sents):
            output_sents_dict[str(ref_id)] = [output_sent]
        # print('output_sents_dict:', len(output_sents_dict))
        target_sents_dict = {}
        for ref_id, target_sent in zip(list(range(len(target_sents))), target_sents):
            target_sents_dict[str(ref_id)] = [target_sent]
        # print('target_sent_dict', len(target_sents_dict))
        # It seems change nothing bt PTBTokenizer,emmmmmm还是有点用的
        tokenizer = PTBTokenizer()
        output_sents_dict = tokenizer.tokenize(output_sents_dict)
        target_sents_dict = tokenizer.tokenize(target_sents_dict)
        # print('output_sents_dict after tokenize', len(output_sents_dict))
        # print('target_sents_dict after tokenize', len(target_sents_dict))

        output_reward, output_rewards = rewarder.compute_score(target_sents_dict, output_sents_dict)
        output_rewards = torch.from_numpy(output_rewards).to(device)
        # print('output_rewards:', len(output_rewards))

        beam_output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            num_beams=5,
            num_return_sequences=5,
            max_length=20,
        )
        beam_sents = self.tokenizer.batch_decode(beam_output, skip_special_tokens=True)
        beam_target_sents = []
        for target_sent in target_sents:
            beam_target_sents += [target_sent]*5

        beam_sents_dict = {}
        for idx, beam_sent in zip(list(range(len(beam_sents))), beam_sents):
            beam_sents_dict[str(idx)] = [beam_sent]

        beam_target_sents_dict = {}
        for idx, beam_target_sent in zip(list(range(len(beam_target_sents))), beam_target_sents):
            beam_target_sents_dict[str(idx)] = [beam_target_sent]
        beam_reward, beam_rewards = rewarder.compute_score(beam_target_sents_dict, beam_sents_dict)
        beam_rewards = torch.from_numpy(beam_rewards).to(device)
        beam_rewards = beam_rewards.view(-1, 5)
        reward_baseline = torch.mean(beam_rewards, dim=1)
        # print(output_rewards.size(), reward_baseline.size())
        loss = (output_rewards-reward_baseline)*loss
        loss = loss.mean()
        reslut['loss'] = loss

        return reslut


    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']

        # generate 可以指定num_beams, 以及num_return_sequence(default=1), so here return only 1 sentence for 1 ref_id！
        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )

        # this is a list type, length equal to batch size,
        # e.g.['A giraffe standing in the shade of a tree.','A giraffe standing in the middle of two other giraffes.', ...]
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = []
        i = 0
        for sent in generated_sents:
            tmp = {}
            ref_id = ref_ids[i]
            tmp['ref_id'] = ref_id
            tmp['sent'] = sent
            # 这招在以后遇到需要在dataloader之后append的，都可以用
            tmp_copy = copy.deepcopy(tmp)
            result.append(tmp_copy)
            i = i+1

        return result


from modeling_bart import VLBart
class VLBartREG(VLBart):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=reduce_loss
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result