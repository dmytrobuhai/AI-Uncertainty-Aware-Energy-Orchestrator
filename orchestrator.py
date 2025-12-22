import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from pathlib import Path
from torch_geometric.nn import GATConv

# ============================================================
# NEW ENCODER-DECODER MODEL
# ============================================================
class SystemEncoderDecoder(nn.Module):
    def __init__(
        self,
        num_regions=27,
        num_features=15, 
        hidden_dim=128,
        future_feat_per_step=5, 
        horizon=6
    ):
        super().__init__()
        self.R = num_regions
        self.H = horizon
        self.hidden_dim = hidden_dim
        
        self.reg_compressor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.encoder_gru = nn.GRU(
            input_size=num_regions * 32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.future_encoder = nn.Linear(future_feat_per_step, 16)

        self.decoder_cell = nn.GRUCell(
            input_size=16 + (num_regions * 1), 
            hidden_size=hidden_dim * 2 
        )

        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_regions)
        )

        self.target_idx = 2 

    def forward(self, x_past, x_future):
        B, T, R, F = x_past.shape
        x_compressed = self.reg_compressor(x_past.reshape(B*T*R, F)).view(B, T, R*32)
        _, h_n = self.encoder_gru(x_compressed) 
        h_t = torch.cat([h_n[-2], h_n[-1]], dim=-1) 

        current_deficit = x_past[:, -1, :, self.target_idx] 
        predictions = []
        future_steps = x_future.view(B, self.H, -1) 

        for i in range(self.H):
            f_step = self.future_encoder(future_steps[:, i, :]) 
            decoder_in = torch.cat([current_deficit, f_step], dim=-1)
            h_t = self.decoder_cell(decoder_in, h_t)
            out = self.out_head(h_t) 
            predictions.append(out)
            current_deficit = out 

        return torch.stack(predictions, dim=1)

# ============================================================
# TEMPORAL GAT (RISK MODEL)
# ============================================================
class TemporalGAT(nn.Module):
    def __init__(self, in_dim, gat_hidden=64, gat_heads=4, gru_hidden=64):
        super().__init__()
        self.gat1 = GATConv(in_dim, gat_hidden, heads=gat_heads, concat=True, dropout=0.1)
        self.gat2 = GATConv(gat_hidden * gat_heads, gat_hidden, heads=1, concat=False, dropout=0.1)
        self.gru = nn.GRU(gat_hidden, gru_hidden, batch_first=False)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, edge_index):
        embeds = []
        for t in range(x_seq.shape[0]):
            h = F.elu(self.gat1(x_seq[t], edge_index))
            h = self.gat2(h, edge_index)
            embeds.append(h)
        H = torch.stack(embeds, dim=0)
        out, _ = self.gru(H)
        logits = self.mlp(out[-1]).squeeze(-1)
        return logits

# ============================================================
# ORCHESTRATOR
# ============================================================
class EnergyOrchestrator:
    def __init__(self, model_dir="serialized"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)

        # Завантаження скейлерів
        try:
            self.gru_scaler = joblib.load(self.model_dir / "scaler.pkl")
            self.gnn_scaler = joblib.load(self.model_dir / "risk_gnn_scaler.pkl")
        except FileNotFoundError as e:
            raise RuntimeError(f"Не знайдено файли скейлерів у {model_dir}") from e

        self.regions = [
            "Вінницька","Волинська","Дніпропетровська","Донецька","Житомирська",
            "Закарпатська","Запорізька","Івано-Франківська","Київська","Кіровоградська",
            "Луганська","Львівська","Миколаївська","Одеська","Полтавська",
            "Рівненська","Сумська","Тернопільська","Харківська","Херсонська",
            "Хмельницька","Черкаська","Чернівецька","Чернігівська","Київ",
            "м. Севастополь","АР Крим"
        ]

        self.region_coords = {
            "Вінницька": (49.233, 28.467), "Волинська": (50.733, 25.317),
            "Дніпропетровська": (48.467, 35.033), "Донецька": (48.000, 37.800),
            "Житомирська": (50.250, 28.650), "Закарпатська": (48.617, 22.283),
            "Запорізька": (47.833, 35.133), "Івано-Франківська": (48.917, 24.700),
            "Київська": (50.450, 30.517), "Київ": (50.450, 30.517),
            "Кіровоградська": (48.500, 32.267), "Луганська": (48.567, 39.300),
            "Львівська": (49.833, 24.017), "Миколаївська": (46.967, 31.983),
            "Одеська": (46.467, 30.717), "Полтавська": (49.583, 34.550),
            "Рівненська": (50.617, 26.250), "Сумська": (50.900, 34.783),
            "Тернопільська": (49.550, 25.583), "Харківська": (49.983, 36.217),
            "Херсонська": (46.633, 32.600), "Хмельницька": (49.417, 26.967),
            "Черкаська": (49.433, 32.050), "Чернівецька": (48.283, 25.933),
            "Чернігівська": (51.483, 31.283), "АР Крим": (44.950, 34.100),
            "м. Севастополь": (44.600, 33.533),
        }

        # Ініціалізація та завантаження нової Encoder-Decoder моделі
        self.load_model = SystemEncoderDecoder(
            num_regions=27, 
            num_features=15,
            hidden_dim=128
        ).to(self.device)
        
        # Назва файлу може відрізнятися, переконайтеся що вона вірна
        model_path = self.model_dir / "system_gru_best.pt" 
        self.load_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.load_model.eval()

        # Завантаження Risk GNN
        gnn_ckpt = torch.load(self.model_dir / "risk_gnn_model.pt", map_location=self.device)
        self.gnn_model = TemporalGAT(in_dim=12).to(self.device)
        self.gnn_model.load_state_dict(gnn_ckpt["model_state_dict"])
        self.gnn_model.eval()

        self.edge_index = gnn_ckpt["edge_index"].to(self.device)
        self.node_names = gnn_ckpt["node_names"]
        self.region_to_node = {
            n.replace("REGION::", ""): i 
            for i, n in enumerate(self.node_names) if n.startswith("REGION::")
        }

    def _denorm(self, x):
        """Зворотне масштабування дефіциту"""
        mean = self.gru_scaler.mean_[2]
        std = self.gru_scaler.scale_[2]
        return x * std + mean

    def decision_policy(self, deficit, risk):
        if risk > 0.75: return "CRITICAL_SHUTDOWN", "Критичний структурний ризик. Ізоляція вузла."
        if deficit > 100: return "EMERGENCY_CUTS", "Аварійний дефіцит. Негайні відключення."
        if deficit > 20: return "SCHEDULED_BLACKOUT", "Планові відключення."
        if deficit < 5: return "STABLE", "Система стабільна."
        return "ECONOMY_MODE", "Рекомендовано зменшити споживання."

    @torch.no_grad()
    def run_inference(self, x_past, x_future, gnn_tensor):
        # 1. Scaling для Encoder-Decoder
        B, T, R, F = x_past.shape
        x_past_flat = x_past.reshape(-1, F).cpu().numpy()
        x_past_scaled = self.gru_scaler.transform(x_past_flat)
        x_past_tensor = torch.tensor(x_past_scaled, dtype=torch.float32).view(B, T, R, F).to(self.device)

        # 2. Scaling для Risk GNN
        Days, N, Gf = gnn_tensor.shape
        gnn_flat = gnn_tensor.reshape(-1, Gf).cpu().numpy()
        gnn_scaled = self.gnn_scaler.transform(gnn_flat)
        gnn_tensor_scaled = torch.tensor(gnn_scaled, dtype=torch.float32).view(Days, N, Gf).to(self.device)

        # 3. Model Inference
        # x_future подається як є (B, 30), бо це симуляція
        pred_scaled = self.load_model(x_past_tensor, x_future.to(self.device))[0].cpu().numpy()
        deficits = self._denorm(pred_scaled) # (Horizon, Regions)

        risks = torch.sigmoid(self.gnn_model(gnn_tensor_scaled, self.edge_index)).cpu().numpy()

        # 4. Формування результатів
        report = {}
        for r_idx, r in enumerate(self.regions):
            node_idx = self.region_to_node.get(r, 0)
            risk = float(risks[node_idx])
            sched = []
            for h in range(6):
                d = float(deficits[h, r_idx])
                mode, rec = self.decision_policy(d, risk)
                sched.append({
                    "hour": h+1,
                    "deficit": round(d, 2),
                    "mode": mode,
                    "recommendation": rec
                })
            report[r] = {"strategic_risk_score": risk, "schedule": sched}

        return report, risks, deficits

    def get_balancing_recommendations(self, report, risks, deficits, threshold=5, risk_limit=0.8, min_gen=10):
        donors, receivers = [], []
        
        for i, r in enumerate(self.regions):
            d = deficits[0, i] 
            n_idx = self.region_to_node.get(r, 0)
            risk = float(risks[n_idx])
            
            # Регіон-донор: d < 0, і його профіцит більший за min_gen, а ризик прийнятний
            if d < -min_gen and risk <= risk_limit:
                donors.append({"name": r, "val": abs(d)})
                
            # Регіон-реципієнт: d > threshold
            if d > threshold:
                receivers.append({"name": r, "val": d})

        # Сортування для пріоритетності
        donors = sorted(donors, key=lambda x: x['val'], reverse=True)
        receivers = sorted(receivers, key=lambda x: x['val'], reverse=True)

        recs = []
        for rec in receivers:
            if not donors:
                break
            
            don = donors[0]
            # Передаємо стільки, скільки можемо
            transfer = min(don['val'], rec['val'])
            
            if transfer > 0.5: # ігноруємо мікро-перетоки
                recs.append({
                    "from": don['name'],
                    "to": rec['name'],
                    "amount": round(transfer, 1),
                    "reason": f"Часткове покриття дефіциту ({rec['val']:.1f} МВт) за рахунок резерву."
                })
                
                # Оновлюємо залишок у донора
                don['val'] -= transfer
                if don['val'] < 1: # якщо донор "порожній"
                    donors.pop(0)
                    
        return recs