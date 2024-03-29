import torch.nn as nn
import matplotlib.pyplot as plt
import torch


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()


class PairWiseLoss(nn.Module):
    """Pairwise Loss for Ranking Tasks.

    This PyTorch module computes a pairwise loss for ranking tasks where the goal is to compare two inputs and determine
    which one is "better" than the other. Given two input tensors: `chosen_reward` and `reject_reward`, which should
    contain reward values for the "chosen" and "rejected" options, respectively, this module computes the probability of
    the chosen option being "better" than the rejected option using a sigmoid function, and then takes the negative
    logarithm of that probability to get the loss. The loss is then averaged over the batch dimension and returned as a
    scalar tensor. Note that this module assumes that higher reward values indicate better options.
    """

    def __init__(self):
        super(PairWiseLoss, self).__init__()

    def forward(self, chosen_reward, reject_reward) -> torch.Tensor:
        """Compute pairwise loss.

        Args:
        - chosen_reward: A tensor of shape (batch_size,) containing reward values for the chosen option
        - reject_reward: A tensor of shape (batch_size,) containing reward values for the rejected option

        Returns:
        - loss: A scalar tensor containing the computed pairwise loss
        """

        # Compute probability of the chosen option being better than the rejected option
        probs = torch.sigmoid(chosen_reward - reject_reward)

        # Take the negative logarithm of the probability to get the loss
        log_probs = torch.log(probs)
        loss = -log_probs.mean()

        return loss
