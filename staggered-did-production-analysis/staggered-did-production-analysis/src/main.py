#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Staggered DiD Production Analysis - CLI Application
====================================================

Command-line interface for bakery manufacturing equipment analysis.

Usage:
    python main.py <input_file> --output <output_folder> [options]

Examples:
    python main.py data.xlsx --output results/
    python main.py data.xlsx --output results/ --analysis all
    python main.py data.xlsx --output results/ --analysis survival --cfr-threshold 0.15
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bathtub_analysis import BathtubAnalyzer
from survival_analysis import SurvivalAnalyzer, RightCensoringHandler
from did_analysis import DiDAnalyzer
from reservoir_computing import RightCensoringImputer, MTBFPredictor


# ============================================================
# CLI Main Class
# ============================================================

class ProductionAnalysisCLI:
    """
    Command-line interface for production analysis.
    """
    
    def __init__(self, args):
        self.args = args
        self.df = None
        self.results = {}
        self.log_messages = []
    
    def log(self, message: str):
        """Log a message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        self.log_messages.append(formatted)
    
    def load_data(self) -> bool:
        """Load input data."""
        try:
            self.log(f"Loading: {self.args.input}")
            
            # Try Survival_Data sheet first
            try:
                self.df = pd.read_excel(self.args.input, sheet_name='Survival_Data')
            except:
                self.df = pd.read_excel(self.args.input)
            
            self.log(f"Loaded {len(self.df)} records")
            return True
            
        except Exception as e:
            self.log(f"Error loading data: {e}")
            return False
    
    def run_bathtub(self):
        """Run bathtub curve analysis."""
        self.log("\n--- Bathtub Curve Analysis ---")
        
        try:
            analyzer = BathtubAnalyzer(cfr_threshold=self.args.cfr_threshold)
            
            # Find columns
            prod_col = self._find_column(['Cumulative_Production', 'production', 'Units'])
            mtbf_col = self._find_column(['MTBF', 'mtbf', 'Mean_Time_Between_Failures'])
            event_col = self._find_column(['Event', 'event', 'Failure'])
            
            if not prod_col or not mtbf_col:
                self.log("Warning: Required columns not found")
                return
            
            result = analyzer.analyze(
                production=self.df[prod_col].values,
                mtbf=self.df[mtbf_col].values,
                event=self.df[event_col].values if event_col else None,
                equipment_id='All',
                equipment_type='Bakery'
            )
            
            if result:
                self.log(f"  Weibull beta: {result.overall_beta:.2f}" if not pd.isna(result.overall_beta) else "  Weibull beta: N/A")
                self.log(f"  Overall phase: {result.overall_phase}")
                self.log(f"  DFR ends at: {result.dfr_end:,.0f}" if result.dfr_end else "  DFR end: Not detected")
                self.log(f"  CFR ends at: {result.cfr_end:,.0f}" if result.cfr_end else "  CFR end: Not detected")
                
                self.results['Bathtub_Analysis'] = pd.DataFrame([{
                    'Equipment': result.equipment_id,
                    'DFR_End': result.dfr_end,
                    'CFR_End': result.cfr_end,
                    'Weibull_Beta': result.overall_beta,
                    'Overall_Phase': result.overall_phase,
                    'Interpretation': result.interpretation
                }])
        
        except Exception as e:
            self.log(f"Bathtub analysis error: {e}")
    
    def run_survival(self):
        """Run survival analysis."""
        self.log("\n--- Survival Analysis ---")
        
        try:
            analyzer = SurvivalAnalyzer()
            
            # Find columns
            duration_col = self._find_column(['MTBF', 'mtbf', 'duration', 'TTF'])
            event_col = self._find_column(['Event', 'event', 'Failure'])
            
            if not duration_col:
                self.log("Warning: Duration column not found")
                return
            
            duration = self.df[duration_col].dropna().values
            event = self.df[event_col].values if event_col else np.ones(len(duration))
            
            # Prepare covariates
            cov_cols = []
            for col in ['treatment', 'Treatment', 'line', 'Line']:
                if col in self.df.columns:
                    cov_cols.append(col)
            
            covariates = pd.get_dummies(self.df[cov_cols], drop_first=True) if cov_cols else None
            
            # Cox model
            cox_result = analyzer.fit_cox(duration, event, covariates)
            
            if cox_result:
                self.log(f"  Cox Concordance: {cox_result.concordance_index:.3f}")
                self.log(f"  N Events: {cox_result.n_events}/{cox_result.n_observations}")
                self.results['Cox_Results'] = cox_result.summary_df
            
            # AFT comparison
            comparison = analyzer.compare_models(duration, event, covariates)
            if len(comparison) > 0:
                self.results['AFT_Comparison'] = comparison
                best = comparison.iloc[0]['Distribution']
                self.log(f"  Best AFT model: {best}")
        
        except Exception as e:
            self.log(f"Survival analysis error: {e}")
    
    def run_did(self):
        """Run DiD analysis."""
        self.log("\n--- DiD Analysis ---")
        
        try:
            analyzer = DiDAnalyzer(cfr_threshold=self.args.cfr_threshold)
            analyzer.df = self.df.copy()
            
            # Ensure treatment variables exist
            if 'treatment' not in analyzer.df.columns:
                treat_col = self._find_column(['Treatment', 'Treated', 'Is_Treated'])
                if treat_col:
                    analyzer.df['treatment'] = analyzer.df[treat_col]
                else:
                    analyzer.df['treatment'] = 0
            
            if 'post' not in analyzer.df.columns:
                post_col = self._find_column(['Post', 'After', 'Is_Post'])
                if post_col:
                    analyzer.df['post'] = analyzer.df[post_col]
                else:
                    analyzer.df['post'] = 0
            
            analyzer.df['treatment_post'] = analyzer.df['treatment'] * analyzer.df['post']
            
            # Find outcome
            outcome_col = self._find_column(['MTBF', 'mtbf', 'RF_MTBF'])
            
            if not outcome_col:
                self.log("Warning: Outcome column not found")
                return
            
            # Raw DiD
            raw_results = analyzer.calc_raw_did(outcome_col=outcome_col)
            
            if raw_results:
                did_rows = []
                for r in raw_results:
                    self.log(f"  {r.interpretation}")
                    did_rows.append({
                        'Metric': r.metric,
                        'Group': r.group,
                        'DiD_Effect': r.did_effect,
                        'DiD_Effect_Pct': r.did_effect_pct,
                        'P_Value': r.p_value,
                        'Significant': r.is_significant
                    })
                self.results['DiD_Results'] = pd.DataFrame(did_rows)
            
            # TWFE
            twfe_result = analyzer.run_twfe(outcome_col=outcome_col)
            if twfe_result:
                self.log(f"  TWFE coefficient: {twfe_result.coefficient:.2f} (p={twfe_result.p_value:.4f})")
                self.results['TWFE_Results'] = pd.DataFrame([{
                    'Coefficient': twfe_result.coefficient,
                    'Std_Error': twfe_result.std_error,
                    'P_Value': twfe_result.p_value,
                    'R_Squared': twfe_result.r_squared
                }])
        
        except Exception as e:
            self.log(f"DiD analysis error: {e}")
    
    def run_rc(self):
        """Run Reservoir Computing analysis."""
        self.log("\n--- Reservoir Computing ---")
        
        try:
            imputer = RightCensoringImputer(
                n_reservoir=self.args.reservoir_nodes,
                spectral_radius=0.95
            )
            
            # Find columns
            mtbf_col = self._find_column(['MTBF', 'mtbf'])
            event_col = self._find_column(['Event', 'event'])
            prod_col = self._find_column(['Cumulative_Production', 'production'])
            
            if not mtbf_col or not prod_col:
                self.log("Warning: Required columns not found")
                return
            
            mtbf = self.df[mtbf_col].values
            event = self.df[event_col].values if event_col else np.ones(len(mtbf))
            production = self.df[prod_col].values
            
            # Remove NaN
            valid = ~(np.isnan(mtbf) | np.isnan(production))
            
            if valid.sum() < 20:
                self.log("Warning: Insufficient data for RC")
                return
            
            result = imputer.fit_transform(
                mtbf=mtbf[valid],
                event=event[valid].astype(int),
                production=production[valid]
            )
            
            n_imputed = result.imputation_mask.sum()
            self.log(f"  Observations imputed: {n_imputed}")
            
            if n_imputed > 0:
                avg_adj = (result.imputed_values - result.original_values)[result.imputation_mask].mean()
                self.log(f"  Average adjustment: {avg_adj:.2f}")
            
            self.results['RC_Imputation'] = pd.DataFrame({
                'Original': result.original_values,
                'Imputed': result.imputed_values,
                'Is_Imputed': result.imputation_mask,
                'Confidence': result.confidence
            })
        
        except Exception as e:
            self.log(f"RC analysis error: {e}")
    
    def _find_column(self, candidates: list) -> str:
        """Find first matching column name."""
        for col in candidates:
            if col in self.df.columns:
                return col
        return None
    
    def save_results(self):
        """Save results to Excel."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.args.input))[0]
        output_file = os.path.join(self.args.output, f"{base_name}_analysis_{timestamp}.xlsx")
        
        os.makedirs(self.args.output, exist_ok=True)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary
            summary = pd.DataFrame({
                'Item': ['Date', 'Input', 'CFR_Threshold', 'Reservoir_Nodes'],
                'Value': [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.args.input,
                    self.args.cfr_threshold,
                    self.args.reservoir_nodes
                ]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Results
            for name, df in self.results.items():
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    df.to_excel(writer, sheet_name=name[:31], index=False)
        
        # Save log
        log_file = os.path.join(self.args.output, f"log_{timestamp}.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_messages))
        
        self.log(f"\nResults saved: {output_file}")
        self.log(f"Log saved: {log_file}")
        
        return output_file
    
    def run(self):
        """Run the analysis."""
        self.log("=" * 60)
        self.log("Staggered DiD Production Analysis")
        self.log("=" * 60)
        
        if not self.load_data():
            return False
        
        # Run selected analyses
        analyses = self.args.analysis.lower().split(',')
        
        if 'all' in analyses or 'bathtub' in analyses:
            self.run_bathtub()
        
        if 'all' in analyses or 'survival' in analyses:
            self.run_survival()
        
        if 'all' in analyses or 'did' in analyses:
            self.run_did()
        
        if 'all' in analyses or 'rc' in analyses:
            self.run_rc()
        
        # Save
        self.save_results()
        
        self.log("\n" + "=" * 60)
        self.log("Analysis complete!")
        self.log("=" * 60)
        
        return True


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Staggered DiD Production Analysis for Bakery Manufacturing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data.xlsx --output results/
  python main.py data.xlsx --output results/ --analysis all
  python main.py data.xlsx --output results/ --analysis survival,did
  python main.py data.xlsx --output results/ --cfr-threshold 0.20
        """
    )
    
    parser.add_argument('input', help='Input Excel file path')
    parser.add_argument('--output', '-o', required=True, help='Output folder path')
    parser.add_argument('--analysis', '-a', default='all',
                       help='Analysis types: all, bathtub, survival, did, rc (comma-separated)')
    parser.add_argument('--cfr-threshold', type=float, default=0.15,
                       help='CFR classification threshold (default: 0.15)')
    parser.add_argument('--reservoir-nodes', type=int, default=100,
                       help='Number of reservoir nodes (default: 100)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Run analysis
    cli = ProductionAnalysisCLI(args)
    success = cli.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
