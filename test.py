import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser

def calculate_persistent_entropy(point_cloud: np.ndarray) -> float:
    """
    Tính Persistent Entropy cho các lỗ hổng 0D (H0) từ đám mây điểm.
    Entropy càng cao = Cấu trúc hình học càng hỗn loạn / nhiễu.
    Entropy đột ngột thay đổi = Có sự biến đổi hình thái không gian trạng thái.
    """
    # Chạy thuật toán Vietoris-Rips filtration
    result = ripser(point_cloud, maxdim=1)
    diagrams = result["dgms"]
    
    # Lấy thông tin thời điểm "sinh ra" (birth) và "mất đi" (death) của các lỗ hổng H0
    h0_dgram = diagrams[0]
    
    # Loại bỏ điểm vô hạn (infinity)
    h0_dgram = h0_dgram[h0_dgram[:, 1] != np.inf]
    if len(h0_dgram) == 0:
        return 0.0
    
    # Độ dài tuổi thọ của từng lỗ hổng trong không gian
    lifespans = h0_dgram[:, 1] - h0_dgram[:, 0]
    total_lifespan = np.sum(lifespans)
    
    if total_lifespan == 0:
        return 0.0
    
    # Tính xác suất phân bố tuổi thọ và Entropy (Shannon Entropy)
    probs = lifespans / total_lifespan
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(entropy)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHẠY SLIDING WINDOW VÀ TRÍCH XUẤT FEATURE TDA
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 15  # Cửa sổ 15 ngày
tda_entropy_list = [np.nan] * WINDOW_SIZE  # Các ngày đầu chưa đủ window sẽ nhận NaN

print("\n=== 2. ĐANG TÍNH TDA FEATURE QUA CỬA SỔ TRƯỢT ===")

for i in range(WINDOW_SIZE, len(df)):
    # Lấy cửa sổ 15 ngày giá
    window_data = df["Close"].iloc[i - WINDOW_SIZE : i].values
    
    # --- TAKENS' EMBEDDING ---
    # Chuyển chuỗi 1D [p_t, p_{t+1}, ...] thành Point Cloud 2D: [ (p_t, p_{t+1}), (p_{t+1}, p_{t+2}), ... ]
    # Giúp tái tạo lại "không gian trạng thái" (Phase Space Reconstruction) của giá
    point_cloud = np.column_stack([window_data[:-1], window_data[1:]])
    
    # Tính Entropy hình học
    entropy = calculate_persistent_entropy(point_cloud)
    tda_entropy_list.append(entropy)

df["TDA_Entropy"] = tda_entropy_list

# Fillna ngày đầu bằng 0.0
df["TDA_Entropy"] = df["TDA_Entropy"].fillna(0.0)

print("Đã hoàn thành! Kết quả 5 dòng cuối:")
print(df.tail())


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRỰC QUAN HÓA (VISUALIZATION)
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Đồ thị 1: Chuỗi giá Close
ax1.plot(df["Date"], df["Close"], color="blue", label="Close Price")
ax1.axvline(x=df["Date"].iloc[50], color="red", linestyle="--", label="Regime Change (Shock)")
ax1.set_title("1. Simulated Stock Price (Close)")
ax1.set_ylabel("Price ($)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Đồ thị 2: TDA Persistent Entropy
ax2.plot(df["Date"], df["TDA_Entropy"], color="green", label="TDA Persistent Entropy")
ax2.axvline(x=df["Date"].iloc[50], color="red", linestyle="--", label="Regime Change (Shock)")
ax2.set_title("2. Extracted TDA Feature (Persistent Entropy)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Entropy Value")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()