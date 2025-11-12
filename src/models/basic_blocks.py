import torch
import torch.nn as nn
from src.models.utils import grad_reverse

class OutcomeHead(nn.Module):
    def __init__(self, hidden_size, fc_hidden_size, dim_outcome=1, dim_outcome_disc=0):
        super().__init__()
        self.dim_outcome = dim_outcome
        self.dim_outcome_disc = dim_outcome_disc
        self.dim_outcome_cont = dim_outcome - dim_outcome_disc
        self.linear1 = nn.Linear(hidden_size, fc_hidden_size)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(fc_hidden_size, dim_outcome)
        self.trainable_params = ['linear1', 'linear2']
    
    def build_outcome(self, hr):
        """
        hr: hidden representation of patient state, shape (b, hr_size)
        returns: outcome, shape (b, dim_outcome)
        """
        x = self.elu(self.linear1(hr))
        outcome = self.linear2(x)
        #first dim_outcome_disc elements need to be transformed to [0, 1] through sigmoid function
        if self.dim_outcome_disc > 0:
            prob = torch.sigmoid(outcome[:, :self.dim_outcome_disc])
            outcome = torch.concat([prob, outcome[:, self.dim_outcome_disc:]], dim = -1)
        return outcome

class PropensityHead(nn.Module):
    def __init__(self, hidden_size, fc_hidden_size, treatment_type = 'disc'):
        """
        treatment_type: 'disc' or 'cont'
        if treatment_type is 'disc', the output is a sigmoid probability
        if treatment_type is 'cont', the output is the mean and log-variance of the conditional density function
        """
        super().__init__()
        assert treatment_type in ['disc', 'cont']
        self.treatment_type = treatment_type
        self.linear1 = nn.Linear(hidden_size, fc_hidden_size)
        self.elu = nn.ELU()
        if treatment_type == 'disc':
            self.linear2 = nn.Linear(fc_hidden_size, 1)
        else:
            self.linear2 = nn.Linear(fc_hidden_size, 2)
        self.trainable_params = ['linear1', 'linear2']
    
    def build_parameter(self, hr):
        """
        build parameter of the statistical distribution
        hr: hidden representation of patient state, shape (b, .., hr_size)
        returns: propensity, shape (b, .. , 1) or (mu, log_var), where both have shape (b, .., 1)
        """
        x = self.elu(self.linear1(hr))
        propensity = self.linear2(x)
        if self.treatment_type == 'disc':
            proba = torch.sigmoid(propensity)
            return proba
        else:
            mu, log_var = torch.split(propensity, 1, dim = -1)
            log_var = torch.clamp(log_var, -10., 10.)
            return mu, log_var

class OutcomeHead_GRNN(nn.Module):
    """Used by G_RNN"""

    def __init__(self, seq_hidden_units, hr_size, fc_hidden_units, dim_treatments, dim_outcome):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.hr_size = hr_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome

        self.linear1 = nn.Linear(self.hr_size + self.dim_treatments,
                                 self.fc_hidden_units)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(self.fc_hidden_units, self.dim_outcome)
        self.trainable_params = ['linear1', 'linear2']

    def build_outcome(self, hr, current_treatment):
        x = torch.cat((hr, current_treatment), dim=-1)
        x = self.elu(self.linear1(x))
        outcome = self.linear2(x)
        return outcome
    
class OutcomeHead_GNET(nn.Module):
    def __init__(self, input_size, fc_hidden_size, dim_outcome=1):
        super().__init__()
        self.dim_outcome = dim_outcome
        self.linear1 = nn.Linear(input_size, fc_hidden_size)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(fc_hidden_size, dim_outcome)
        self.trainable_params = ['linear1', 'linear2']
    
    def build_outcome(self, input):
        """
        returns: outcome, shape (b, dim_outcome)
        """
        x = self.elu(self.linear1(input))
        outcome = self.linear2(x)
        return outcome
    

class BRTreatmentOutcomeHead(nn.Module):
    """Used by Causal Transformer"""

    def __init__(self, seq_hidden_units, br_size, fc_hidden_units, dim_treatments, dim_outcome, alpha=0.0, update_alpha=True,
                 balancing='grad_reverse'):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.br_size = br_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome
        self.alpha = alpha if not update_alpha else 0.0
        self.alpha_max = alpha
        self.balancing = balancing

        self.linear1 = nn.Linear(self.seq_hidden_units, self.br_size)
        self.elu1 = nn.ELU()

        self.linear2 = nn.Linear(self.br_size, self.fc_hidden_units)
        self.elu2 = nn.ELU()
        self.linear3 = nn.Linear(self.fc_hidden_units, self.dim_treatments)

        self.linear4 = nn.Linear(self.br_size + self.dim_treatments, self.fc_hidden_units)
        self.elu3 = nn.ELU()
        self.linear5 = nn.Linear(self.fc_hidden_units, self.dim_outcome)

        self.treatment_head_params = ['linear2', 'linear3']

    def build_treatment(self, br, detached=False):
        if detached:
            br = br.detach()

        if self.balancing == 'grad_reverse':
            br = grad_reverse(br, self.alpha)

        br = self.elu2(self.linear2(br))
        treatment = self.linear3(br)  # Softmax is encapsulated into F.cross_entropy()
        return treatment

    def build_outcome(self, br, current_treatment):
        x = torch.cat((br, current_treatment), dim=-1)
        x = self.elu3(self.linear4(x))
        outcome = self.linear5(x)
        return outcome

    def build_br(self, seq_output):
        br = self.elu1(self.linear1(seq_output))
        return br