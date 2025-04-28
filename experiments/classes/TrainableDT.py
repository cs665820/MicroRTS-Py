import torch
import torch.nn.functional as F
from transformers import DecisionTransformerModel


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        filtered_kwargs = {k: v for k,
                           v in kwargs.items() if k != "invalid_action_masks"}
        output = super().forward(**filtered_kwargs)

        action_preds = output[1]
        action_targets = kwargs["actions"]
        invalid_action_masks = kwargs["invalid_action_masks"]
        states = kwargs["states"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        state_dim = states.shape[2]

        action_preds = action_preds.reshape(-1,
                                            act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1,
                                                act_dim)[attention_mask.reshape(-1) > 0]
        invalid_action_masks = invalid_action_masks.reshape(
            -1, act_dim)[attention_mask.reshape(-1) > 0]
        states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]

        action_targets = action_targets.clone()
        action_targets = action_targets.reshape(-1, 78)  # 78 action encodings
        invalid_action_masks = invalid_action_masks.reshape(-1, 78)
        invalid_action_masks[invalid_action_masks == 0] = -9e8
        action_preds = action_preds.reshape(-1, 78)  # 78 action encodings
        states = states.reshape(-1, 6)  # 6 state encodings

        probs = torch.zeros_like(
            action_preds, dtype=torch.float32, device=action_preds.device)

        slices = [
            (0, 6),    # action type
            (6, 10),   # move parameter
            (10, 14),  # harvest parameter
            (14, 18),  # return parameter
            (18, 22),  # produce direction parameter
            (22, 29),  # produce type parameter
            (29, 78)   # relative attack position
        ]

        for start, end in slices:
            probs[:, start:end] = F.softmax(
                action_preds[:, start:end] + invalid_action_masks[:, start:end], dim=1)

        # mask units that are not owned by the agent
        mask = ~(invalid_action_masks[:, 0:6] == -9e8).all(dim=1)
        probs = probs[mask]
        action_targets = action_targets[mask]

        loss = F.cross_entropy(probs, action_targets)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        filtered_kwargs = {k: v for k,
                           v in kwargs.items() if k != "invalid_action_mask"}
        output = super().forward(**filtered_kwargs)
        action_preds = output[1]
        act_dim = kwargs["actions"].shape[2]
        mask = kwargs["invalid_action_mask"]
        mask = mask.reshape(-1, 78)
        mask[mask == 0] = -9e8
        action_pred = action_preds.reshape(-1, act_dim)[-1]  # next action
        action_pred = action_pred.reshape(-1, 78)  # 78 action encodings
        action_pred += mask  # mask invalid actions

        slices = [
            (0, 6),    # action type
            (6, 10),   # move parameter
            (10, 14),  # harvest parameter
            (14, 18),  # return parameter
            (18, 22),  # produce direction parameter
            (22, 29),  # produce type parameter
            (29, 78)   # relative attack position
        ]

        for start, end in slices:
            probs = F.softmax(action_pred[:, start:end], dim=1)
            action_pred[:, start:end] = torch.nn.functional.one_hot(
                torch.argmax(probs, dim=1), num_classes=end-start
            )

        return action_pred.flatten()
