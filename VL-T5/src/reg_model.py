import torch
import torch.nn as nn
import copy
# from memory_profiler import profile


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