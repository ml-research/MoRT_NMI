import torch


class ClassificationMCMHead(torch.nn.Module):
    """Classification Head for transformer encoders"""

    def __init__(self, base_mode_tokenizer, mcm):
        super().__init__()
        self.mcm = mcm
        self.base_mode_tokenizer = base_mode_tokenizer

    def convert_to_mcm_tokens(self, base_mode_hidden):
        message = self.base_mode_tokenizer.decode(base_mode_hidden, skip_special_tokens=True)
        return message

    def forward(self, message):
        # decode
        # eot token: 50256
        mcm_res, _, _ = self.mcm.bias(message)
        logits = mcm_res[0]
        logits = torch.nn.functional.tanh(logits)
        return logits
