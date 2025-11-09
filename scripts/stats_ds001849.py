#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical analysis for DDS parameters from TMS-EEG data (ds001849)
Focuses on site and condition effects using mixed-effects models
Publication-quality version with enhanced visualizations - Outlier exclusion and violin plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

# Statistics packages
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
from scipy.stats import f_oneway, ttest_rel, ttest_ind

print("=== DDS Parameter Statistical Analysis ===")

# Set publication-quality style parameters
def set_publication_style():
    """Set matplotlib and seaborn parameters for publication-quality figures"""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'font.family': 'Arial',
        'mathtext.fontset': 'stix',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.6
    })
    sns.set_style("whitegrid", {
        'grid.linestyle': ':',
        'grid.alpha': 0.3,
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8
    })

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers from specified columns using IQR method
    Returns cleaned dataframe and outlier indices
    """
    df_clean = df.copy()
    outlier_indices = []
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        # Calculate Q1, Q3, and IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Find outliers
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        outlier_indices.extend(outliers.index.tolist())
        
        print(f"  {col}: {len(outliers)} outliers removed ({lower_bound:.2f} to {upper_bound:.2f})")
    
    # Remove duplicates and get unique outlier indices
    outlier_indices = list(set(outlier_indices))
    
    if outlier_indices:
        df_clean = df_clean.drop(outlier_indices)
        print(f"Total outliers removed: {len(outlier_indices)}")
    else:
        print("No outliers found")
    
    return df_clean, outlier_indices

def load_data(csv_path):
    """Load DDS parameters and prepare for analysis"""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert to categorical with meaningful order
    df['site'] = pd.Categorical(df['site'], categories=['m1', 'dlpfc', 'ppc'])
    df['cond'] = pd.Categorical(df['cond'], categories=['active', 'sham'])
    df['subject'] = pd.Categorical(df['subject'])
    df['window'] = pd.Categorical(df['window'])
    
    print(f"Total data: {len(df)} rows, {df['subject'].nunique()} subjects")
    print(f"Windows: {df['window'].unique().tolist()}")
    
    # Check data completeness
    subject_counts = df.groupby(['window', 'site', 'cond'])['subject'].nunique()
    print("\nNumber of unique subjects by window, site and condition:")
    print(subject_counts)
    
    return df

def descriptive_statistics(df, window_label=""):
    """Compute descriptive statistics for DDS parameters"""
    if window_label:
        print(f"\n{'='*60}")
        print(f"DESCRIPTIVE STATISTICS - {window_label.upper()}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)
    
    dds_params = ['A1', 'gamma1', 'f1', 'A2', 'gamma2', 'f2', 'R2', 'RMSE_uV']
    
    for param in dds_params:
        print(f"\n--- {param} ---")
        # Check if parameter exists in dataframe
        if param not in df.columns:
            print(f"Parameter {param} not found in data")
            continue
            
        stats_df = df.groupby(['site', 'cond'])[param].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        print(stats_df)

def plot_dds_parameters_violin(df, output_dir, window='15-80ms'):
    """Create publication-quality visualization of DDS parameters using violin plots"""
    print(f"\nGenerating publication-quality DDS parameter VIOLIN plots for {window} window...")
    print(f"Output directory: {output_dir.absolute()}")
    
    dds_params = ['A1', 'gamma1', 'f1', 'A2', 'gamma2', 'f2']
    # Filter out parameters that don't exist in the data
    dds_params = [p for p in dds_params if p in df.columns]
    
    if not dds_params:
        print("No DDS parameters found for plotting")
        return
    
    # Remove outliers for better visualization
    #print(f"\nRemoving outliers for {window} window...")
    #df_clean, outlier_indices = remove_outliers_iqr(df, dds_params, factor=1.5)
    
    #print(f"Data after outlier removal: {len(df_clean)} rows (removed {len(outlier_indices)})")
        
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.ravel()
    
    # Publication color palette
    condition_palette = {'active': '#E74C3C', 'sham': '#3498DB'}  # Professional red/blue
    condition_palette_light = {'active': '#FADBD8', 'sham': '#D6EAF8'}  # Light versions
    
    # Parameter units and labels for proper formatting
    param_labels = {
    'A1': 'A1 (µV)',
    'gamma1': 'γ1 (s⁻¹)',
    'f1': 'f1 (Hz)',
    'A2': 'A2 (µV)',
    'gamma2': 'γ2 (s⁻¹)',
    'f2': 'f2 (Hz)'
    }
    
    for i, param in enumerate(dds_params):
        if i >= len(axes):
            break
            
        plt.sca(axes[i])
        
        # Create enhanced violin plot
        violin = sns.violinplot(data=df, x='site', y=param, hue='cond', 
                               palette=condition_palette_light,
                               linewidth=1.5, width=0.8, saturation=0.8,
                               inner='quartile',  # Show quartiles inside violin
                               cut=0,  # Don't cut the violin at the extremes
                               scale='width')  # Scale violins by width
        
        # Add individual points with better styling (optional - can comment out if too crowded)
        stripplot = sns.stripplot(data=df, x='site', y=param, hue='cond',
                                 dodge=True, jitter=True, alpha=0.6, size=3,
                                 palette=condition_palette, linewidth=0.3,
                                 edgecolor='white')
        
        # Enhanced titles and labels
        plt.title(f'{param}', fontweight='bold', pad=20)
        plt.ylabel(param_labels.get(param, param), fontweight='bold')
        plt.xlabel('Stimulation Site', fontweight='bold')
        
        # Improve x-axis labels
        site_labels = ['M1', 'DLPFC', 'PPC']
        plt.gca().set_xticklabels(site_labels, fontweight='bold')
        
        # Remove duplicate legends and create custom legend
        if i > 0:
            plt.gca().get_legend().remove()
        else:
            # Create custom legend for violin plots
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=condition_palette_light['active'], edgecolor='black', 
                      label='Active', alpha=0.8),
                Patch(facecolor=condition_palette_light['sham'], edgecolor='black', 
                      label='Sham', alpha=0.8),
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=condition_palette['active'], 
                          markersize=6, markeredgecolor='white', linewidth=0, 
                          label='Active (individual)'),
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=condition_palette['sham'], 
                          markersize=6, markeredgecolor='white', linewidth=0, 
                          label='Sham (individual)')
            ]
            plt.legend(handles=legend_elements, title='Condition', 
                      frameon=True, fancybox=True, shadow=True,
                      loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
        
        # Remove top and right spines for cleaner look
        sns.despine(top=True, right=True)
    
    # Main title for the entire figure
    fig.suptitle(f'DDS Parameters by Site and Condition - {window}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save with publication quality
    filename = f'dds_parameters_violin_by_site_condition_{window.replace("-", "_")}_publication.png'
    full_path = output_dir / filename
    plt.savefig(full_path, dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved: {full_path}")
    plt.close()
    
    # Enhanced R² and RMSE plot with violin plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if 'R2' in df.columns:
        sns.violinplot(data=df, x='site', y='R2', hue='cond', 
                    palette=condition_palette_light, linewidth=1.5,
                    ax=ax1, width=0.8, inner='quartile',
                    cut=0, scale='width')
        sns.stripplot(data=df, x='site', y='R2', hue='cond',
                     dodge=True, jitter=True, alpha=0.6, size=3,
                     palette=condition_palette, ax=ax1, linewidth=0.3,
                     edgecolor='white')
        ax1.set_title('Model R² by Site and Condition', fontweight='bold')
        ax1.set_ylabel('R²', fontweight='bold')
        ax1.set_xlabel('Stimulation Site', fontweight='bold')
        ax1.set_xticklabels(['M1', 'DLPFC', 'PPC'], fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.legend().remove()
        sns.despine(top=True, right=True, ax=ax1)
    
    if 'RMSE_uV' in df.columns:
        sns.violinplot(data=df, x='site', y='RMSE_uV', hue='cond',
                    palette=condition_palette_light, linewidth=1.5,
                    ax=ax2, width=0.8, inner='quartile',
                    cut=0, scale='width')
        sns.stripplot(data=df, x='site', y='RMSE_uV', hue='cond',
                     dodge=True, jitter=True, alpha=0.6, size=3,
                     palette=condition_palette, ax=ax2, linewidth=0.3,
                     edgecolor='white')
        ax2.set_title('Model RMSE by Site and Condition', fontweight='bold')
        ax2.set_ylabel('RMSE (µV)', fontweight='bold')
        ax2.set_xlabel('Stimulation Site', fontweight='bold')
        ax2.set_xticklabels(['M1', 'DLPFC', 'PPC'], fontweight='bold')
        ax2.legend().remove()
        sns.despine(top=True, right=True, ax=ax2)
    
    # Create unified legend
    handles, labels = ax2.get_legend_handles_labels()
    # Only keep unique labels (remove duplicates from stripplot)
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    fig.legend(unique_handles, unique_labels, 
               title='Condition', loc='upper center', 
               bbox_to_anchor=(0.5, 0.05), ncol=2,
               frameon=True, fancybox=True, shadow=True)
    
    fig.suptitle(f'Model Fit Metrics - {window}', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make space for legend
    
    # Save with publication quality
    filename = f'model_fit_metrics_violin_{window.replace("-", "_")}_publication.png'
    full_path = output_dir / filename
    plt.savefig(full_path, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {full_path}")
    plt.close()
    
    return df

def create_summary_plot(df, output_dir):
    """Create a comprehensive summary plot showing key findings"""
    print("\nCreating comprehensive summary plot...")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Select key parameters for summary
    key_params = ['gamma1','gamma2', 'f1','f2','A2']
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    
    # Professional color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    condition_palette = {'active': '#E74C3C', 'sham': '#3498DB'}
    
    for i, param in enumerate(key_params):
        if param not in df.columns:
            continue
            
        plt.sca(axes[i])
        
        # Calculate means and standard errors
        summary = df.groupby(['site', 'cond'])[param].agg(['mean', 'sem']).reset_index()
        
        # Create bar plot with error bars
        x_pos = np.arange(len(['m1', 'dlpfc', 'ppc']))
        width = 0.35
        
        for j, cond in enumerate(['active', 'sham']):
            cond_data = summary[summary['cond'] == cond]
            means = cond_data['mean'].values
            sems = cond_data['sem'].values
            
            bars = plt.bar(x_pos + j * width, means, width, 
                          label=cond.capitalize(),
                          color=condition_palette[cond],
                          alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add error bars
            plt.errorbar(x_pos + j * width, means, yerr=sems, 
                        fmt='none', c='black', capsize=5, linewidth=1)
            
            # Add value labels on bars
            for k, (mean, sem) in enumerate(zip(means, sems)):
                plt.text(x_pos[k] + j * width, mean + sem + (0.05 * max(means)), 
                        f'{mean:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Formatting
        # Formatting
        plt.title(f'{param} by Site and Condition', fontweight='bold', pad=20)

        # Add proper unit labels based on parameter type
        if param in ['A1', 'A2']:
            plt.ylabel(f'{param} (µV)', fontweight='bold')
        elif param in ['gamma1', 'gamma2']:
            plt.ylabel(f'{param} (s⁻¹)', fontweight='bold')
        elif param in ['f1', 'f2']:
            plt.ylabel(f'{param} (Hz)', fontweight='bold')
        else:
            plt.ylabel(param, fontweight='bold')
    
        plt.xlabel('Stimulation Site', fontweight='bold')
        plt.xticks(x_pos + width/2, ['M1', 'DLPFC', 'PPC'], fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        sns.despine(top=True, right=True)
        
        if i == 0:
            plt.legend(frameon=True, fancybox=True, shadow=True)
    
    fig.suptitle('Summary of Key DDS Parameters', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save summary plot
    filename = 'dds_parameters_summary_publication.png'
    full_path = output_dir / filename
    plt.savefig(full_path, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {full_path}")
    plt.close()

def mixed_effects_analysis(df, window_label=""):
    """Perform mixed-effects modeling for each DDS parameter"""
    if window_label:
        print(f"\n{'='*60}")
        print(f"MIXED-EFFECTS MODEL ANALYSIS - {window_label.upper()}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("MIXED-EFFECTS MODEL ANALYSIS")
        print("="*60)
    
    dds_params = ['A1', 'gamma1', 'f1', 'A2', 'gamma2', 'f2']
    # Filter out parameters that don't exist or have issues
    dds_params = [p for p in dds_params if p in df.columns]
    
    results = []
    
    for param in dds_params:
        print(f"\n--- Mixed-effects model for {param} ---")
        
        # Check for sufficient data and variability
        param_data = df[param].dropna()
        if len(param_data) < 10 or param_data.std() < 1e-10:
            print(f"Insufficient data or zero variance for {param}, skipping")
            continue
            
        # Model: parameter ~ site + condition + site:condition + (1|subject)
        formula = f"{param} ~ C(site) + C(cond) + C(site):C(cond)"
        
        try:
            model = mixedlm(formula, df, groups=df['subject'])
            result = model.fit(method='lbfgs', maxiter=1000)
            print(result.summary())
            
            # Extract p-values for fixed effects
            p_values = result.pvalues
            for effect in ['C(site)', 'C(cond)', 'C(site):C(cond)']:
                if effect in p_values:
                    results.append({
                        'parameter': param,
                        'effect': effect,
                        'p_value': p_values[effect],
                        'significant': p_values[effect] < 0.05
                    })
                    
        except Exception as e:
            print(f"Error fitting model for {param}: {e}")
            # Try simpler model without interaction
            try:
                print("Trying simpler model without interaction...")
                formula_simple = f"{param} ~ C(site) + C(cond)"
                model = mixedlm(formula_simple, df, groups=df['subject'])
                result = model.fit(method='lbfgs', maxiter=1000)
                print(result.summary())
                
                p_values = result.pvalues
                for effect in ['C(site)', 'C(cond)']:
                    if effect in p_values:
                        results.append({
                            'parameter': param,
                            'effect': effect,
                            'p_value': p_values[effect],
                            'significant': p_values[effect] < 0.05
                        })
            except Exception as e2:
                print(f"Also failed with simpler model: {e2}")
    
    # Create results dataframe
    if results:
        results_df = pd.DataFrame(results)
        
        # Apply FDR correction
        rejected, pvals_corrected, _, _ = multipletests(
            results_df['p_value'], alpha=0.05, method='fdr_bh'
        )
        results_df['p_value_fdr'] = pvals_corrected
        results_df['significant_fdr'] = rejected
        
        print("\nMixed-effects model results with FDR correction:")
        print(results_df.round(4))
        
        return results_df
    else:
        print("No successful model fits")
        return None
def debug_subject_matching(df, window='15-80ms'):
    """Debug why subject matching is failing"""
    window_data = df[df['window'] == window]
    
    print("=== SUBJECT MATCHING DEBUG ===")
    for cond in ['active', 'sham']:
        print(f"\n{cond.upper()} condition:")
        for site1, site2 in [('m1', 'dlpfc'), ('m1', 'ppc'), ('dlpfc', 'ppc')]:
            subs1 = set(window_data[(window_data['site'] == site1) & (window_data['cond'] == cond)]['subject'])
            subs2 = set(window_data[(window_data['site'] == site2) & (window_data['cond'] == cond)]['subject'])
            common = subs1.intersection(subs2)
            
            print(f"  {site1} vs {site2}: {len(subs1)} vs {len(subs2)} subjects, {len(common)} common")
            if len(common) < 10:  # If less than half the expected subjects
                print(f"    {site1} subjects: {sorted(subs1)}")
                print(f"    {site2} subjects: {sorted(subs2)}")
                print(f"    Common: {sorted(common)}")


def site_pairwise_comparisons(df, window_label=""):
    """Pairwise comparisons between sites for each DDS parameter"""
    if window_label:
        print(f"\n{'='*60}")
        print(f"PAIRWISE SITE COMPARISONS - {window_label.upper()}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("PAIRWISE SITE COMPARISONS")
        print("="*60)
    
    dds_params = ['A1', 'gamma1', 'f1', 'A2', 'gamma2', 'f2']
    dds_params = [p for p in dds_params if p in df.columns]
    
    sites = ['m1', 'dlpfc', 'ppc']
    comparisons = [('m1', 'dlpfc'), ('m1', 'ppc'), ('dlpfc', 'ppc')]
    # Check data completeness for this specific dataframe
    for param in ['A1', 'gamma1', 'f1', 'A2', 'gamma2', 'f2']:
        print(f"\n{param} availability:")
        for site in ['m1', 'dlpfc', 'ppc']:
            for cond in ['active', 'sham']:
                count = df[(df['site'] == site) & (df['cond'] == cond)][param].notna().sum()
                print(f"  {site}-{cond}: {count}")
    all_results = []
    
    for param in dds_params:
        print(f"\n--- {param} ---")
        
        for cond in ['active', 'sham']:
            print(f"  {cond.upper()} condition:")
            
            for site1, site2 in comparisons:
                # Extract data for this comparison
                data1 = df[(df['site'] == site1) & (df['cond'] == cond)][param].dropna()
                data2 = df[(df['site'] == site2) & (df['cond'] == cond)][param].dropna()
                
                # Ensure we have paired data
                common_subjects = set(df[(df['site'] == site1) & (df['cond'] == cond)]['subject']) & \
                                set(df[(df['site'] == site2) & (df['cond'] == cond)]['subject'])
                
                if len(common_subjects) < 2:
                    print(f"    {site1} vs {site2}: Insufficient paired data ({len(common_subjects)} common subjects)")
                    continue
                
                # Get paired data
                paired_data1 = []
                paired_data2 = []
                for subj in common_subjects:
                    val1 = df[(df['subject'] == subj) & (df['site'] == site1) & (df['cond'] == cond)][param].values
                    val2 = df[(df['subject'] == subj) & (df['site'] == site2) & (df['cond'] == cond)][param].values
                    if len(val1) > 0 and len(val2) > 0:
                        paired_data1.append(val1[0])
                        paired_data2.append(val2[0])
                
                if len(paired_data1) > 1 and len(paired_data2) > 1:
                    # Paired t-test
                    try:
                        t_stat, p_val = ttest_rel(paired_data1, paired_data2)
                        effect_size = np.abs(np.mean(paired_data1) - np.mean(paired_data2)) / np.sqrt((np.std(paired_data1)**2 + np.std(paired_data2)**2)/2)
                        
                        all_results.append({
                            'parameter': param,
                            'condition': cond,
                            'comparison': f"{site1}-vs-{site2}",
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'effect_size': effect_size,
                            'mean_diff': np.mean(paired_data1) - np.mean(paired_data2),
                            'n_pairs': len(paired_data1)
                        })
                        
                        sig_symbol = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                        print(f"    {site1} vs {site2}: t({len(paired_data1)-1})={t_stat:.3f}, p={p_val:.4f} {sig_symbol}")
                        
                    except Exception as e:
                        print(f"    {site1} vs {site2}: Error - {e}")
    
    # Create results dataframe and apply FDR correction PER PARAMETER family
    if all_results:
        comp_df = pd.DataFrame(all_results)
    
        # Apply FDR correction separately for each parameter and condition
        comp_df['p_value_fdr'] = np.nan
        comp_df['significant_fdr'] = False
    
        for param in comp_df['parameter'].unique():
            for cond in comp_df['condition'].unique():
                mask = (comp_df['parameter'] == param) & (comp_df['condition'] == cond)
                if mask.sum() > 1:  # Only apply FDR if multiple tests for this parameter-condition
                    rejected, pvals_corrected, _, _ = multipletests(
                        comp_df.loc[mask, 'p_value'], alpha=0.05, method='fdr_bh'
                    )
                    comp_df.loc[mask, 'p_value_fdr'] = pvals_corrected
                    comp_df.loc[mask, 'significant_fdr'] = rejected
                elif mask.sum() == 1:
                    # Single test - use raw p-value
                    comp_df.loc[mask, 'p_value_fdr'] = comp_df.loc[mask, 'p_value']
                    comp_df.loc[mask, 'significant_fdr'] = comp_df.loc[mask, 'p_value'] < 0.05
        
        print("\nPairwise comparisons with FDR correction:")
        print(comp_df.round(4))
        
        return comp_df
    
    return None

def condition_comparisons(df, window_label=""):
    """Compare active vs sham within each site"""
    if window_label:
        print(f"\n{'='*60}")
        print(f"ACTIVE vs SHAM COMPARISONS BY SITE - {window_label.upper()}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("ACTIVE vs SHAM COMPARISONS BY SITE")
        print("="*60)
    
    dds_params = ['A1', 'gamma1', 'f1', 'A2', 'gamma2', 'f2']
    dds_params = [p for p in dds_params if p in df.columns]
    
    sites = ['m1', 'dlpfc', 'ppc']
    
    all_results = []
    
    for param in dds_params:
        print(f"\n--- {param} ---")
        
        for site in sites:
            # Get common subjects across conditions for this site
            active_subjects = set(df[(df['site'] == site) & (df['cond'] == 'active')]['subject'])
            sham_subjects = set(df[(df['site'] == site) & (df['cond'] == 'sham')]['subject'])
            common_subjects = active_subjects & sham_subjects
            
            if len(common_subjects) < 2:
                print(f"  {site}: Insufficient paired data ({len(common_subjects)} common subjects)")
                continue
            
            # Get paired data
            active_data = []
            sham_data = []
            for subj in common_subjects:
                active_val = df[(df['subject'] == subj) & (df['site'] == site) & (df['cond'] == 'active')][param].values
                sham_val = df[(df['subject'] == subj) & (df['site'] == site) & (df['cond'] == 'sham')][param].values
                if len(active_val) > 0 and len(sham_val) > 0:
                    active_data.append(active_val[0])
                    sham_data.append(sham_val[0])
            
            if len(active_data) > 1 and len(sham_data) > 1:
                # Paired t-test
                try:
                    t_stat, p_val = ttest_rel(active_data, sham_data)
                    effect_size = np.abs(np.mean(active_data) - np.mean(sham_data)) / np.sqrt((np.std(active_data)**2 + np.std(sham_data)**2)/2)
                    
                    all_results.append({
                        'parameter': param,
                        'site': site,
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'effect_size': effect_size,
                        'mean_diff': np.mean(active_data) - np.mean(sham_data),
                        'n_pairs': len(active_data)
                    })
                    
                    sig_symbol = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"  {site}: t({len(active_data)-1})={t_stat:.3f}, p={p_val:.4f} {sig_symbol}")
                    print(f"    Active: {np.mean(active_data):.4f} ± {np.std(active_data):.4f}")
                    print(f"    Sham: {np.mean(sham_data):.4f} ± {np.std(sham_data):.4f}")
                    
                except Exception as e:
                    print(f"  {site}: Error - {e}")
    
    # Apply FDR correction PER PARAMETER family
    if all_results:
        cond_df = pd.DataFrame(all_results)
    
        # Apply FDR correction separately for each parameter
        cond_df['p_value_fdr'] = np.nan
        cond_df['significant_fdr'] = False
    
        for param in cond_df['parameter'].unique():
            param_mask = cond_df['parameter'] == param
            if param_mask.sum() > 1:  # Only apply FDR if multiple tests for this parameter
                rejected, pvals_corrected, _, _ = multipletests(
                    cond_df.loc[param_mask, 'p_value'], alpha=0.05, method='fdr_bh'
                )
                cond_df.loc[param_mask, 'p_value_fdr'] = pvals_corrected
                cond_df.loc[param_mask, 'significant_fdr'] = rejected
            else:
                # Single test - use raw p-value
                cond_df.loc[param_mask, 'p_value_fdr'] = cond_df.loc[param_mask, 'p_value']
                cond_df.loc[param_mask, 'significant_fdr'] = cond_df.loc[param_mask, 'p_value'] < 0.05
        
        print("\nCondition comparisons with FDR correction:")
        print(cond_df.round(4))
        
        return cond_df
    
    return None

def analyze_both_windows(df, output_dir):
    """Run analysis for both early and late windows"""
    print("=== COMPARATIVE DDS ANALYSIS: EARLY vs LATE WINDOWS ===\n")
    
    windows = ['15-80ms', '80-200ms']
    all_results = {}
    
    for window in windows:
        print(f"\n{'='*80}")
        print(f"ANALYZING {window.upper()} WINDOW")
        print(f"{'='*80}")
        
        # Filter for current window
        window_df = df[df['window'] == window].copy()
        
        # Only analyze if we have valid data
        if len(window_df) == 0:
            print(f"No data found for {window}")
            continue
            
        print(f"Data: {len(window_df)} rows, {window_df['subject'].nunique()} subjects")
        
        # Create window-specific output directory
        window_output_dir = output_dir / f"{window.replace('-', '_')}"
        window_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analyses
        descriptive_statistics(window_df, window)
        
        # Use violin plots instead of box plots
        df_clean = plot_dds_parameters_violin(window_df, window_output_dir, window)
        
        # Use cleaned data for statistical analyses
        mixed_results = mixed_effects_analysis(window_df, window)
        pairwise_results = site_pairwise_comparisons(window_df, window)
        condition_results = condition_comparisons(window_df, window)
        
        # Store results
        all_results[window] = {
            'data': window_df,
            'data_clean': df_clean,
            'mixed_results': mixed_results,
            'pairwise_results': pairwise_results,
            'condition_results': condition_results
        }
        
        # Save results to files
        if mixed_results is not None:
            mixed_results.to_csv(window_output_dir / 'mixed_effects_results.csv', index=False)
        
        if pairwise_results is not None:
            pairwise_results.to_csv(window_output_dir / 'pairwise_site_comparisons.csv', index=False)
        
        if condition_results is not None:
            condition_results.to_csv(window_output_dir / 'condition_comparisons.csv', index=False)
    
    return all_results

def window_comparison_analysis(df, output_dir):
    """Compare DDS parameters between early and late windows"""
    print("\n" + "="*80)
    print("EARLY vs LATE WINDOW COMPARISONS")
    print("="*80)
    
    dds_params = ['A1', 'gamma1', 'f1', 'A2', 'gamma2', 'f2', 'R2', 'RMSE_uV']
    sites = ['m1', 'dlpfc', 'ppc']
    conditions = ['active', 'sham']
    
    comparison_results = []
    
    for param in dds_params:
        print(f"\n--- {param} ---")
        
        for site in sites:
            for cond in conditions:
                # Get common subjects across windows for this site and condition
                early_subjects = set(df[(df['window'] == '15-80ms') & (df['site'] == site) & (df['cond'] == cond)]['subject'])
                late_subjects = set(df[(df['window'] == '80-200ms') & (df['site'] == site) & (df['cond'] == cond)]['subject'])
                common_subjects = early_subjects & late_subjects
                
                if len(common_subjects) < 2:
                    print(f"  {site}-{cond}: Insufficient paired data ({len(common_subjects)} common subjects)")
                    continue
                
                # Get paired data
                early_data = []
                late_data = []
                for subj in common_subjects:
                    early_val = df[(df['subject'] == subj) & (df['window'] == '15-80ms') & 
                                 (df['site'] == site) & (df['cond'] == cond)][param].values
                    late_val = df[(df['subject'] == subj) & (df['window'] == '80-200ms') & 
                                (df['site'] == site) & (df['cond'] == cond)][param].values
                    if len(early_val) > 0 and len(late_val) > 0:
                        early_data.append(early_val[0])
                        late_data.append(late_val[0])
                
                if len(early_data) > 1 and len(late_data) > 1:
                    # Paired t-test
                    try:
                        t_stat, p_val = ttest_rel(early_data, late_data)
                        effect_size = np.abs(np.mean(early_data) - np.mean(late_data)) / np.sqrt((np.std(early_data)**2 + np.std(late_data)**2)/2)
                        
                        comparison_results.append({
                            'parameter': param,
                            'site': site,
                            'condition': cond,
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'effect_size': effect_size,
                            'mean_diff': np.mean(early_data) - np.mean(late_data),
                            'early_mean': np.mean(early_data),
                            'late_mean': np.mean(late_data),
                            'n_pairs': len(early_data)
                        })
                        
                        sig_symbol = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                        print(f"  {site}-{cond}: t={t_stat:.3f}, p={p_val:.4f} {sig_symbol}")
                        print(f"    Early: {np.mean(early_data):.4f} ± {np.std(early_data):.4f}")
                        print(f"    Late:  {np.mean(late_data):.4f} ± {np.std(late_data):.4f}")
                        
                    except Exception as e:
                        print(f"  {site}-{cond}: Error - {e}")
    
   
    # Apply FDR correction PER PARAMETER family
    if comparison_results:
        comp_df = pd.DataFrame(comparison_results)
    
        # Apply FDR correction separately for each parameter
        comp_df['p_value_fdr'] = np.nan
        comp_df['significant_fdr'] = False
    
        for param in comp_df['parameter'].unique():
            param_mask = comp_df['parameter'] == param
            if param_mask.sum() > 1:  # Only apply FDR if multiple tests for this parameter
                rejected, pvals_corrected, _, _ = multipletests(
                    comp_df.loc[param_mask, 'p_value'], alpha=0.05, method='fdr_bh'
                )
                comp_df.loc[param_mask, 'p_value_fdr'] = pvals_corrected
                comp_df.loc[param_mask, 'significant_fdr'] = rejected
            else:
                # Single test - use raw p-value
                comp_df.loc[param_mask, 'p_value_fdr'] = comp_df.loc[param_mask, 'p_value']
                comp_df.loc[param_mask, 'significant_fdr'] = comp_df.loc[param_mask, 'p_value'] < 0.05
        
        print("\nWindow comparisons with FDR correction:")
        print(comp_df.round(4))
        
        # Save results
        comp_df.to_csv(output_dir / 'window_comparisons.csv', index=False)
        
        # Summary of significant window effects
        sig_effects = comp_df[comp_df['significant_fdr']]
        if not sig_effects.empty:
            print("\nSignificant window effects (FDR-corrected):")
            for _, row in sig_effects.iterrows():
                direction = "decreased" if row['mean_diff'] > 0 else "increased"
                print(f"  {row['parameter']} at {row['site']}-{row['condition']}: {direction} from early to late window (p = {row['p_value_fdr']:.4f})")
        
        return comp_df
    
    return None
def check_fit_quality_impact(df, window='15-80ms'):
    """Check if fit quality metrics explain missing data"""
    window_data = df[df['window'] == window]
    
    print("=== FIT QUALITY ANALYSIS ===")
    for site in ['m1', 'dlpfc', 'ppc']:
        for cond in ['active', 'sham']:
            site_data = window_data[(window_data['site'] == site) & (window_data['cond'] == cond)]
            
            # Check R2 values for fits that succeeded vs overall
            valid_fits = site_data[site_data['A1'].notna()]
            total_trials = len(site_data)
            valid_trials = len(valid_fits)
            
            print(f"\n{site}-{cond}: {valid_trials}/{total_trials} valid fits")
            
            if valid_trials > 0:
                print(f"  R2: {valid_fits['R2'].mean():.3f} ± {valid_fits['R2'].std():.3f}")
                print(f"  RMSE: {valid_fits['RMSE_uV'].mean():.3f} ± {valid_fits['RMSE_uV'].std():.3f}")
            
            # Check if there's a pattern in which subjects failed
            failed_subjects = set(site_data[site_data['A1'].isna()]['subject'])
            if failed_subjects:
                print(f"  Failed for subjects: {sorted(failed_subjects)}")

# Check if there's code that filters parameters based on thresholds
def check_parameter_filtering(df):
    """Check if parameters are being filtered by thresholds"""
    print("=== PARAMETER FILTERING CHECK ===")
    
    # Common filtering criteria to check
    thresholds = {
        'R2': 0.3,    # Minimum R²
        'RMSE_uV': 50, # Maximum RMSE
        'gamma1': (80, 400),  # Range for gamma1
        'gamma2': (15, 100),  # Range for gamma2
    }
    
    for param, threshold in thresholds.items():
        if param in df.columns:
            if isinstance(threshold, tuple):
                # Range check
                filtered = df[(df[param] < threshold[0]) | (df[param] > threshold[1])]
            else:
                # Single threshold check
                if param == 'R2':
                    filtered = df[df[param] < threshold]
                else:  # RMSE_uV
                    filtered = df[df[param] > threshold]
            
            if len(filtered) > 0:
                print(f"{param}: {len(filtered)} rows would be filtered")
                print(f"  Affected sites: {filtered['site'].value_counts().to_dict()}")


def main():
    # Set publication style at the beginning
    set_publication_style()
    
    # Configuration - use absolute paths for clarity
    current_dir = Path(__file__).parent if __file__ in locals() else Path.cwd()
    data_dir = current_dir / '../derivatives/dds_ds001849'
    output_dir = current_dir / '../stats_results'
    
    print(f"Current directory: {current_dir.absolute()}")
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir.absolute()}")
    
    # Find the latest DDS parameters file
    csv_files = list(data_dir.glob('*dds*all_windows.csv'))
    if not csv_files:
        print(f"No DDS parameter file found in {data_dir.absolute()}!")
        print("Available files:", list(data_dir.glob('*')))
        return
    
    csv_path = csv_files[0]
    print(f"Using file: {csv_path.absolute()}")
    
    # Load all data
    df_all = load_data(csv_path)
    # Add this diagnostic right after loading the data
    print("=== DATA LOADING DEBUG ===")
    print(f"Total rows loaded: {len(df_all)}")
    print(f"Subjects: {df_all['subject'].nunique()}")

    # Check if data is being filtered during loading
    print("\nData distribution by site and condition:")
    print(df_all.groupby(['site', 'cond']).size())
    # Verify window filtering
    df_15_80 = df_all[df_all['window'] == '15-80ms']
    print(f"15-80ms data: {len(df_15_80)} rows")
    print("Subjects in 15-80ms window:")
    print(df_15_80.groupby(['site', 'cond'])['subject'].nunique())
    # Check for any data cleaning steps
    print("\nMissing values in loaded data:")
    print(df_all.isnull().sum())
    if len(df_all) == 0:
        print("No data found for analysis!")
        return
    
    # Run analysis for both windows
    window_results = analyze_both_windows(df_all, output_dir)
    
    # Create comprehensive summary plot
    create_summary_plot(df_all, output_dir)
    
    # Compare windows directly
    window_comparison_analysis(df_all, output_dir)
    
    print("\n" + "="*80)
    print("COMPLETE ANALYSIS FINISHED")
    print("="*80)
    print(f"Results saved to: {output_dir.absolute()}")
    print("\nExpected VIOLIN plot files:")
    print(f"- {output_dir.absolute()}/15_80ms/dds_parameters_violin_by_site_condition_15_80ms_publication.png")
    print(f"- {output_dir.absolute()}/15_80ms/model_fit_metrics_violin_15_80ms_publication.png")
    print(f"- {output_dir.absolute()}/80_200ms/dds_parameters_violin_by_site_condition_80_200ms_publication.png")
    print(f"- {output_dir.absolute()}/80_200ms/model_fit_metrics_violin_80_200ms_publication.png")
    print(f"- {output_dir.absolute()}/dds_parameters_summary_publication.png")
    # Run the debug
    debug_subject_matching(df_all, '15-80ms')
    check_fit_quality_impact(df_all, '15-80ms')
    check_parameter_filtering(df_all)
if __name__ == "__main__":
    main()
