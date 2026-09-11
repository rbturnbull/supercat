import plotly.graph_objects as go
import pandas as pd


BICUBIC = 0.24954
HR = 0.257536
LR = 0.250432
porosity_file = "carbonate2D-SRx4-valid-3300-porosity.txt"


BICUBIC = 0.187088
HR = 0.195444
LR = 0.184448
porosity_file = "carbonate2D-SRx4-valid-3400-porosity.txt"

BICUBIC = 0.266356
HR = 0.281944
LR = 0.268672
porosity_file = "carbonate2D-SRx4-valid-3500-porosity.txt"


df = pd.read_csv(porosity_file)

fig = go.Figure()
fig.add_trace(go.Histogram(x=df["porosity"], name="Supercat Diffusion 2D", histnorm="probability"))

markers = [
    (BICUBIC, "Bicubic"),
    (HR, "HR"),
    (LR, "LR"),
    (df["porosity"].mean(), "Mean"),
]

for x_value, label in markers:
    fig.add_vline(x=x_value, line_dash="dash", line_color="black")
    fig.add_annotation(
        x=x_value,
        y=1,
        xref="x",
        yref="paper",
        text=f"{label}<br>{x_value:.5g}",
        showarrow=False,
        yshift=+30,
    )

fig.update_layout(title="Porosity Distribution of Supercat Diffusion 2D", xaxis_title="Porosity", yaxis_title="Probability")

output_path = porosity_file.replace(".txt", ".png")
fig.write_image(output_path, scale=2)
print(f"Porosity distribution plot saved to {output_path}")
