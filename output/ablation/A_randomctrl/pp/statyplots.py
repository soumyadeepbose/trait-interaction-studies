import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Define paths relative to 'ablation/A_randomctrl/pp/'
    current_csv = 'generations_scored.csv' 
    role_csv = os.path.join('..', '..', 'A_role', 'pp', 'generations_scored.csv')
    
    # 2. Load the current folder's CSV (A_randomctrl/pp)
    try:
        df_randomctrl = pd.read_csv(current_csv)
    except FileNotFoundError:
        print(f"Error: '{current_csv}' not found in the current directory.")
        print("Make sure you are running this script inside 'ablation/A_randomctrl/pp/'.")
        sys.exit(1)
        
    # 3. Load the sibling folder's CSV (A_role/pp)
    try:
        df_role = pd.read_csv(role_csv)
    except FileNotFoundError:
        print(f"Error: Could not find the role CSV at expected path: {role_csv}")
        sys.exit(1)

    # 4. Isolate the 3 specific subsets and tag them
    # Group 1: baseline from A_role
    g1 = df_role[df_role['mode'] == 'baseline'].copy()
    g1['group'] = 'A_role (baseline)'
    
    # Group 2: ablate_head from A_role
    g2 = df_role[df_role['mode'] == 'ablate_head'].copy()
    g2['group'] = 'A_role (ablate_head)'
    
    # Group 3: ablate_head from A_randomctrl
    g3 = df_randomctrl[df_randomctrl['mode'] == 'ablate_head'].copy()
    g3['group'] = 'A_randomctrl (ablate_head)'

    # 5. Combine the subsets
    combined_df = pd.concat([g1, g2, g3], ignore_index=True)

    if combined_df.empty:
        print("Error: No data matched your target filters. Check 'mode' column values.")
        sys.exit(1)

    # 6. Calculate means and plot
    metrics = ['coherence', 'evilness', 'adherence']
    mean_scores = combined_df.groupby('group')[metrics].mean()

    # Enforce clear visual order on the plot
    desired_order = ['A_role (baseline)', 'A_role (ablate_head)', 'A_randomctrl (ablate_head)']
    existing_order = [g for g in desired_order if g in mean_scores.index]
    mean_scores = mean_scores.reindex(existing_order)

    # 7. Render Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(11, 6))
    mean_scores.T.plot(kind='bar', width=0.7, edgecolor='black', ax=ax)

    # Styling
    ax.set_title('Mean Metrics Comparison Across Runs', fontsize=14, pad=15)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Mean Values', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title='Subsets', fontsize=10, title_fontsize=11)
    
    plt.tight_layout()

    # Save and display
    output_filename = 'cross_folder_comparison.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Combined plot saved as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    main()