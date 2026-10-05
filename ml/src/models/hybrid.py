"""
Hybrid Architecture: GraphSAGE Structural Embeddings -> XGBoost Gradient Boosted Trees

To avoid target leakage:
  1. GNN embeddings for training XGBoost are generated strictly out-of-fold (5-fold CV on train set).
  2. The final GNN is trained on the full train split and evaluated on val/test graph snapshots.
"""
from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
import xgboost as xgb

from src.eval.temporal_split import SplitResult
from src.models.gnn import GraphSAGE
from src.train.train_gnn import create_pyg_data, train_single_gnn


class GraphSAGE_XGBoost_Hybrid:
    def __init__(
        self,
        gnn_kwargs: Optional[Dict[str, Any]] = None,
        xgb_kwargs: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        self.gnn_kwargs = gnn_kwargs or {"hidden_channels": 128}
        self.xgb_kwargs = xgb_kwargs or {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 150,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": -1,
        }
        self.random_state = random_state
        self.xgb_model = xgb.XGBClassifier(**self.xgb_kwargs, random_state=random_state)
        self.gnn_model: Optional[GraphSAGE] = None

    def fit(self, split: SplitResult, graph: dict, gnn_epochs: int = 80):
        """Fit hybrid pipeline with out-of-fold GNN embeddings."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        train_data = create_pyg_data(split, graph, "train", local_only=False)
        
        # 1. Train final GNN model on full train split
        self.gnn_model = GraphSAGE(in_channels=train_data.x.shape[1], **self.gnn_kwargs)
        self.gnn_model = train_single_gnn(
            self.gnn_model, train_data, val_data=None, lr=0.01, weight_decay=1e-5, epochs=gnn_epochs
        )

        # 2. Out-of-fold embeddings to train XGBoost without target leakage
        labeled_idx = np.where(train_data.mask.numpy())[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        hidden_dim = self.gnn_kwargs.get("hidden_channels", 128)
        oof_embeddings = np.zeros((len(labeled_idx), hidden_dim), dtype=np.float32)

        for fold, (f_train_idx, f_val_idx) in enumerate(kf.split(labeled_idx)):
            fold_train_mask = train_data.mask.clone()
            fold_train_mask[labeled_idx[f_val_idx]] = False

            fold_model = GraphSAGE(in_channels=train_data.x.shape[1], **self.gnn_kwargs)
            optimizer = torch.optim.Adam(fold_model.parameters(), lr=0.01, weight_decay=1e-5)

            for epoch in range(gnn_epochs):
                fold_model.train()
                optimizer.zero_grad()
                out = fold_model(train_data.x, train_data.edge_index).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(
                    out[fold_train_mask], train_data.y[fold_train_mask].float()
                )
                loss.backward()
                optimizer.step()

            fold_model.eval()
            with torch.no_grad():
                emb = fold_model.get_embeddings(train_data.x, train_data.edge_index)
                oof_embeddings[f_val_idx] = emb[labeled_idx[f_val_idx]].cpu().numpy()

        # 3. Train XGBoost on [Local Features + Out-of-Fold GNN Embeddings]
        X_train_local = train_data.x[labeled_idx, :94].cpu().numpy()
        X_train_hybrid = np.concatenate([X_train_local, oof_embeddings], axis=1)
        y_train = train_data.y[labeled_idx].cpu().numpy()

        scale_pos_weight = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
        self.xgb_model.set_params(scale_pos_weight=scale_pos_weight)
        self.xgb_model.fit(X_train_hybrid, y_train)

    def predict_proba(self, split: SplitResult, graph: dict, mode: str = "test") -> np.ndarray:
        """Predict illicit probabilities using the fitted hybrid architecture."""
        if self.gnn_model is None:
            raise RuntimeError("GraphSAGE_XGBoost_Hybrid must be fitted before predict_proba.")

        data = create_pyg_data(split, graph, mode, local_only=False)
        self.gnn_model.eval()
        with torch.no_grad():
            emb = self.gnn_model.get_embeddings(data.x, data.edge_index).cpu().numpy()

        X_local = data.x[:, :94].cpu().numpy()
        X_hybrid = np.concatenate([X_local, emb], axis=1)

        probs = self.xgb_model.predict_proba(X_hybrid)[:, 1]
        return probs
