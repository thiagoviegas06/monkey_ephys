import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_diagram():
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Helper function to draw a box
    def draw_box(x, y, w, h, text, color='lightblue', alpha=0.3, fontsize=10):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                       linewidth=1.5, edgecolor='black', facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, weight='bold', wrap=True)

    # Helper function to draw an arrow
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # --- 1. Input ---
    draw_box(35, 92, 30, 6, "Input: SBP Masked + Mask Indicator\n(B, W, C, 2)", color='lightgrey')
    
    # --- 2. Initial Feature Extraction ---
    draw_arrow(50, 92, 50, 86)
    draw_box(30, 80, 40, 6, "Initial Feature Extraction: TCN\n(Temporal Blocks per channel)", color='lightgreen')
    ax.text(72, 83, "Shape: (B*C, 2, W) -> (B, C, W, D)", fontsize=9, style='italic')

    # --- 3. Interleaved Axial Encoder Blocks ---
    # Container for Encoder
    encoder_y_start = 32
    encoder_height = 44
    rect_encoder = patches.Rectangle((20, encoder_y_start), 60, encoder_height, linewidth=2, 
                                     edgecolor='blue', facecolor='none', linestyle='--', alpha=0.5)
    ax.add_patch(rect_encoder)
    ax.text(21, encoder_y_start + encoder_height - 3, "Interleaved Axial Encoder (N layers)", 
            color='blue', fontsize=12, weight='bold')

    # Temporal Mixing
    draw_arrow(50, 80, 50, 72)
    draw_box(25, 64, 50, 8, "Temporal Mixing: Self-Attention across W\n(Reshape B*C, W, D + Positional Encoding)", color='orange')
    
    # Spatial Mixing
    draw_arrow(50, 64, 50, 56)
    draw_box(25, 48, 50, 8, "Spatial Mixing: Self-Attention across C\n(Reshape B*W, C, D + Channel Embeddings)", color='salmon')

    # Loop back arrow
    ax.annotate('', xy=(50, 74), xytext=(75, 52),
                arrowprops=dict(arrowstyle='<-', lw=1.5, color='blue', connectionstyle="arc3,rad=-0.3"))
    ax.text(78, 63, "Repeat N times", color='blue', fontsize=10, rotation=270, weight='bold')

    # --- 4. Asymmetric Decoder ---
    draw_arrow(50, encoder_y_start, 50, 24)
    draw_box(25, 14, 50, 10, "Asymmetric Decoder: Transformer\n(Across Channels for each Time Bin)\nMask Tokens + Time Embeddings", color='purple')
    ax.text(77, 19, "Shape: (B*W, C, D)", fontsize=9, style='italic')

    # --- 5. Output ---
    draw_arrow(50, 14, 50, 8)
    draw_box(35, 2, 30, 6, "Output: Reconstructed SBP\n(B, W, C)", color='lightgrey')

    # Metadata/Labels
    ax.text(5, 95, "Interleaved Axial Transformer Architecture", fontsize=16, weight='bold')
    ax.text(5, 92, "Flow Visualization", fontsize=12)

    plt.tight_layout()
    plt.savefig('axial_attention_flow.png', dpi=300, bbox_inches='tight')
    print("Diagram saved to axial_attention_flow.png")

if __name__ == "__main__":
    create_diagram()
