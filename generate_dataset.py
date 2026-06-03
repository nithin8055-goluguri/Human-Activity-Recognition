import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
csv_path = os.path.join("data", "webcam_har_data.csv")

activities = [
    "SITTING", "STANDING", "WALKING", "WAVING", 
    "SQUATS", "BICEP_CURLS", "JUMPING_JACKS", 
    "RUNNING", "LAYING", "PUSHUPS", "PUNCHING"
]
samples_per_activity = 400

data_rows = []
print("⏳ Building your 11-action dataset rows...")

for activity in activities:
    for _ in range(samples_per_activity):
        coords = np.random.uniform(0.4, 0.6, 132)
        
        if activity == "SITTING":
            coords[92:112] += 0.25   
        elif activity == "LAYING":
            coords[1:132:4] = np.random.uniform(0.8, 0.95, 33) 
        elif activity == "WAVING":
            coords[60:75] -= 0.3  
        elif activity == "BICEP_CURLS":
            coords[40:55] -= 0.2
        elif activity == "JUMPING_JACKS":
            coords[44:60] -= 0.25    
            coords[112:128] += 0.2   
        elif activity == "PUNCHING":
            coords[40:50] += 0.35 
            
        coords = np.clip(coords, 0.0, 1.0)
        data_rows.append(list(coords) + [activity])

columns = [f"coord_{i}" for i in range(132)] + ["label"]
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(csv_path, index=False)

print(f"[✓] Success! Generated {csv_path} with {len(activities)} actions!")
