import torch
import torch.nn as nn
import torch.nn.functional as F

class FuzzyLayer(nn.Module):
    """
    Differentiable Fuzzy Layer with learnable Gaussian membership functions.
    """
    def __init__(self, num_features, num_rules):
        super(FuzzyLayer, self).__init__()
        self.num_features = num_features
        self.num_rules = num_rules
        
        # Learnable parameters for Gaussian membership functions (mu and sigma)
        # Shape: (num_features, num_rules) - each feature has 'num_rules' fuzzy sets
        self.mu = nn.Parameter(torch.randn(num_features, num_rules))
        self.sigma = nn.Parameter(torch.abs(torch.randn(num_features, num_rules)) + 1e-3)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_features)
        Returns:
            membership: Tensor of shape (batch_size, num_features, num_rules)
        """
        # x: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(-1)
        
        # Gaussian membership function: exp(-((x - mu)^2) / (2 * sigma^2))
        # Broadcasting happens here
        numerator = (x - self.mu) ** 2
        denominator = 2 * (self.sigma ** 2) + 1e-6 # epsilon for stability
        
        membership = torch.exp(-numerator / denominator)
        return membership

class FuzzyRuleLayer(nn.Module):
    """
    Learnable Fuzzy Rule Layer.
    Combines antecedents (fuzzy memberships) to form rule firing strengths.
    """
    def __init__(self, num_features, num_rules):
        super(FuzzyRuleLayer, self).__init__()
        self.num_rules = num_rules
        # In a simple T-norm (product) implementation, we might not need weights here 
        # if we assume fully connected rules or specific structure.
        # For a flexible approach, we can use a linear combination or attention.
        # Here we implement a "Product T-Norm" which is standard for differentiable fuzzy systems.

    def forward(self, membership):
        """
        Args:
            membership: (batch_size, num_features, num_rules)
        Returns:
            rule_strength: (batch_size, num_rules)
        """
        # Product T-Norm: AND operation across features for each rule
        # We assume each rule uses one fuzzy set from each feature (simplified ANFIS)
        # Or we can aggregate across features.
        # Let's assume 'num_rules' is the total number of rules and we aggregate features.
        
        # For this implementation, we'll assume the standard ANFIS-like structure where
        # rule_j = mu_1j * mu_2j * ... * mu_nj
        # PROBLEM: With high dimensions (128), product vanishes (0.5^128 -> 0).
        # FIX: Use Mean aggregation (Compensatory AND) or Min.
        # We use Mean for better gradient flow through all features.
        rule_strength = torch.mean(membership, dim=1)
        return rule_strength

class NeuroFuzzyGIMAN(nn.Module):
    """
    Hybrid Neuro-Fuzzy GIMAN Model.
    Wraps a GAT encoder with a Fuzzy Logic Layer for classification/regression.
    """
    def __init__(self, gat_encoder, num_classes=2, num_rules=16):
        super(NeuroFuzzyGIMAN, self).__init__()
        self.gat_encoder = gat_encoder
        
        # Freeze GAT encoder if we want to preserve Phase 8 baseline features
        # for param in self.gat_encoder.parameters():
        #     param.requires_grad = False
            
        # Get output dimension of GAT (assuming it's available in the object)
        # If not, we might need to pass it explicitly. 
        # Based on previous files, hidden_dim was 128.
        self.gat_out_dim = getattr(gat_encoder, 'hidden_dim', 128)
        
        # Fuzzy Layers
        self.fuzzy_layer = FuzzyLayer(self.gat_out_dim, num_rules)
        self.rule_layer = FuzzyRuleLayer(self.gat_out_dim, num_rules)
        
        # Consequent Layer (Takagi-Sugeno type: linear function of inputs)
        # y_j = w_j * x + b_j
        self.consequent_weight = nn.Parameter(torch.randn(num_rules, self.gat_out_dim))
        self.consequent_bias = nn.Parameter(torch.randn(num_rules))
        
        # Final aggregation
        self.output_layer = nn.Linear(num_rules, num_classes)

    def forward(self, data):
        # 1. Get GAT embeddings
        # We assume gat_encoder returns the embedding before the final risk head
        # We might need to modify GIMANSurvivalGAT to return intermediate features
        # or use a hook. For now, let's assume we can call a method or it returns features.
        # If gat_encoder is the full model, we might need to slice it.
        
        # HACK: If gat_encoder is the full Phase 8 model, it returns risk scores.
        # We need the features. Let's assume we extract the 'convs' part or similar.
        # Ideally, we pass the feature vector 'x' through the convs.
        
        x, edge_index = data.x, data.edge_index
        
        # Replicating GAT forward pass (without the head)
        # This depends on the specific instance passed. 
        # If we pass the class instance, we can use its sub-modules.
        for i, (conv, bn) in enumerate(zip(self.gat_encoder.convs[:-1], self.gat_encoder.batch_norms[:-1])):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = self.gat_encoder.dropout(x)
            
        # Final GAT layer (before head)
        x = self.gat_encoder.convs[-1](x, edge_index)
        x = self.gat_encoder.batch_norms[-1](x)
        features = F.elu(x) # (batch, hidden_dim)
        
        # 2. Fuzzy Fuzzification
        membership = self.fuzzy_layer(features) # (batch, hidden, rules)
        
        # 3. Rule Evaluation
        w = self.rule_layer(membership) # (batch, rules)
        
        # Normalize firing strengths
        w_norm = w / (torch.sum(w, dim=1, keepdim=True) + 1e-6)
        
        # 4. Consequent Evaluation (Takagi-Sugeno)
        # rule_output_j = (features . weight_j) + bias_j
        # We need efficient batch computation
        # features: (B, D), weight: (R, D) -> (B, R)
        consequent_out = F.linear(features, self.consequent_weight, self.consequent_bias)
        
        # 5. Defuzzification (Weighted Average)
        # output = sum(w_norm_j * consequent_out_j)
        output_features = w_norm * consequent_out # (B, R)
        
        # Final classification/regression
        logits = self.output_layer(output_features)
        
        return logits, w_norm # Return weights for interpretability

class MultiTaskNeuroFuzzyGIMAN(nn.Module):
    """
    Multi-Task GIMAN:
    1. SAA Classification (Neuro-Fuzzy Head)
    2. Survival Prediction (GAT-Cox Head)
    """
    def __init__(self, gat_encoder, num_classes=2, num_rules=32):
        super(MultiTaskNeuroFuzzyGIMAN, self).__init__()
        self.gat_encoder = gat_encoder
        
        # Neuro-Fuzzy Head for Classification
        self.gat_out_dim = getattr(gat_encoder, 'hidden_dim', 128)
        self.fuzzy_layer = FuzzyLayer(self.gat_out_dim, num_rules)
        self.rule_layer = FuzzyRuleLayer(self.gat_out_dim, num_rules)
        self.consequent_weight = nn.Parameter(torch.randn(num_rules, self.gat_out_dim))
        self.consequent_bias = nn.Parameter(torch.randn(num_rules))
        self.output_layer = nn.Linear(num_rules, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 1. Shared GAT Encoder (Feature Extraction)
        # We need to manually run GAT layers to get features for Fuzzy Head
        # AND run the full GAT forward for Risk Head (which expects data object usually, 
        # but GIMANSurvivalGAT.forward takes data and returns risk).
        # To avoid re-computation, we should refactor or just run partial.
        
        # Let's extract features first (Shared Backbone)
        for i, (conv, bn) in enumerate(zip(self.gat_encoder.convs[:-1], self.gat_encoder.batch_norms[:-1])):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = self.gat_encoder.dropout(x)
            
        # Final GAT layer (before heads)
        x = self.gat_encoder.convs[-1](x, edge_index)
        x = self.gat_encoder.batch_norms[-1](x)
        features = F.elu(x) # (batch, hidden_dim)
        
        # 2. Task A: Survival Risk (Cox Head)
        # The original risk head takes 'features'
        risk_scores = self.gat_encoder.risk_head(features).squeeze(-1)
        
        # 3. Task B: SAA Classification (Neuro-Fuzzy Head)
        membership = self.fuzzy_layer(features)
        w = self.rule_layer(membership)
        w_norm = w / (torch.sum(w, dim=1, keepdim=True) + 1e-6)
        consequent_out = F.linear(features, self.consequent_weight, self.consequent_bias)
        output_features = w_norm * consequent_out
        logits = self.output_layer(output_features)
        
        return logits, risk_scores, w_norm
