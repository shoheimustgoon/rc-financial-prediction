#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Staggered DiD Production Analysis - GUI Application
====================================================

A comprehensive analysis tool for bakery manufacturing equipment reliability.
Features:
- Bathtub curve analysis (DFR/CFR/IFR phases)
- Survival analysis (Cox/AFT models)
- DiD analysis for intervention effects
- Reservoir Computing for time series

Author: shoheimustgoon
Version: 1.0.0
"""

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

# Check for tkinter
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
    HAS_TK = True
except ImportError:
    HAS_TK = False
    print("Error: tkinter not available. Please run in CLI mode.")

# Import analysis modules
from bathtub_analysis import BathtubAnalyzer, plot_bathtub_curve
from survival_analysis import SurvivalAnalyzer, RightCensoringHandler
from did_analysis import DiDAnalyzer
from reservoir_computing import EchoStateNetwork, RightCensoringImputer

# Check for matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Check for optional libraries
HAS_LIFELINES = False
HAS_STATSMODELS = False
try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    pass

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    pass


# ============================================================
# Constants
# ============================================================

APP_TITLE = "Staggered DiD Production Analysis"
APP_VERSION = "1.0.0"
WINDOW_SIZE = "1100x850"

# Column mappings for different data formats
COLUMN_MAPPINGS = {
    'equipment_id': ['Equipment_ID', 'equipment_id', 'Machine_ID', 'Line_Equipment'],
    'error_datetime': ['Error_DateTime', 'error_datetime', 'Failure_Date', 'Timestamp'],
    'mtbf': ['MTBF', 'mtbf', 'Mean_Time_Between_Failures', 'TTF'],
    'production': ['Cumulative_Production', 'production', 'Total_Output', 'Units_Produced'],
    'event': ['Event', 'event', 'Failure', 'Is_Failure'],
    'line': ['Line', 'line', 'Production_Line', 'Area']
}


# ============================================================
# Main GUI Application
# ============================================================

class ProductionAnalysisGUI:
    """
    Main GUI application for production analysis.
    """
    
    def __init__(self):
        if not HAS_TK:
            raise RuntimeError("tkinter not available")
        
        self.root = tk.Tk()
        self.root.title(f"{APP_TITLE} v{APP_VERSION}")
        self.root.geometry(WINDOW_SIZE)
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.cfr_threshold = tk.DoubleVar(value=0.15)
        self.n_reservoir = tk.IntVar(value=100)
        
        # Analysis options
        self.do_bathtub = tk.BooleanVar(value=True)
        self.do_survival = tk.BooleanVar(value=True)
        self.do_did = tk.BooleanVar(value=True)
        self.do_rc = tk.BooleanVar(value=True)
        
        # Build UI
        self._build_ui()
        
        # Log buffer
        self.log_messages = []
    
    def _build_ui(self):
        """Build the user interface."""
        # Main frame with scrollbar
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Padding
        pad = {'padx': 10, 'pady': 5}
        
        # === Header ===
        header_frame = ttk.Frame(scrollable_frame)
        header_frame.pack(fill='x', **pad)
        
        title_label = ttk.Label(
            header_frame,
            text=f"{APP_TITLE}",
            font=('', 16, 'bold')
        )
        title_label.pack(pady=10)
        
        subtitle_label = ttk.Label(
            header_frame,
            text="Bakery Equipment Reliability Analysis\nBathtub Curve | Survival Analysis | DiD | Reservoir Computing",
            font=('', 11),
            justify='center'
        )
        subtitle_label.pack(pady=5)
        
        # === Library Status ===
        lib_frame = ttk.LabelFrame(scrollable_frame, text="Library Status", padding=10)
        lib_frame.pack(fill='x', **pad)
        
        libs = [
            ('lifelines', HAS_LIFELINES, 'Survival Analysis'),
            ('statsmodels', HAS_STATSMODELS, 'DiD Regression'),
            ('matplotlib', HAS_MATPLOTLIB, 'Visualization')
        ]
        
        lib_inner = ttk.Frame(lib_frame)
        lib_inner.pack(fill='x')
        
        for name, available, desc in libs:
            status = '✓' if available else '✗'
            color = 'green' if available else 'red'
            label = ttk.Label(
                lib_inner,
                text=f"{status} {name} ({desc})",
                foreground=color
            )
            label.pack(side='left', padx=15)
        
        # === Input/Output ===
        io_frame = ttk.LabelFrame(scrollable_frame, text="Input / Output", padding=10)
        io_frame.pack(fill='x', **pad)
        
        # Input file
        ttk.Label(io_frame, text="Input Excel File:").pack(anchor='w')
        input_row = ttk.Frame(io_frame)
        input_row.pack(fill='x', pady=5)
        
        ttk.Entry(input_row, textvariable=self.input_path, width=80).pack(side='left', padx=(0, 5))
        ttk.Button(input_row, text="Browse", command=self._browse_input).pack(side='left')
        
        # Output folder
        ttk.Label(io_frame, text="Output Folder:").pack(anchor='w', pady=(10, 0))
        output_row = ttk.Frame(io_frame)
        output_row.pack(fill='x', pady=5)
        
        ttk.Entry(output_row, textvariable=self.output_path, width=80).pack(side='left', padx=(0, 5))
        ttk.Button(output_row, text="Browse", command=self._browse_output).pack(side='left')
        
        # === Analysis Options ===
        options_frame = ttk.LabelFrame(scrollable_frame, text="Analysis Options", padding=10)
        options_frame.pack(fill='x', **pad)
        
        # Checkboxes
        check_row = ttk.Frame(options_frame)
        check_row.pack(fill='x', pady=5)
        
        ttk.Checkbutton(check_row, text="Bathtub Curve Analysis", variable=self.do_bathtub).pack(side='left', padx=10)
        ttk.Checkbutton(check_row, text="Survival Analysis (Cox/AFT)", variable=self.do_survival).pack(side='left', padx=10)
        ttk.Checkbutton(check_row, text="DiD Analysis", variable=self.do_did).pack(side='left', padx=10)
        ttk.Checkbutton(check_row, text="Reservoir Computing", variable=self.do_rc).pack(side='left', padx=10)
        
        # Parameters
        param_row = ttk.Frame(options_frame)
        param_row.pack(fill='x', pady=10)
        
        # CFR Threshold
        ttk.Label(param_row, text="CFR Threshold:").pack(side='left', padx=(0, 5))
        cfr_scale = ttk.Scale(param_row, from_=0.05, to=0.30, variable=self.cfr_threshold,
                             orient='horizontal', length=150)
        cfr_scale.pack(side='left', padx=(0, 5))
        self.cfr_label = ttk.Label(param_row, text="±0.15")
        self.cfr_label.pack(side='left', padx=(0, 20))
        self.cfr_threshold.trace_add('write', self._update_cfr_label)
        
        # Reservoir nodes
        ttk.Label(param_row, text="Reservoir Nodes:").pack(side='left', padx=(0, 5))
        reservoir_spin = ttk.Spinbox(param_row, from_=20, to=500, textvariable=self.n_reservoir,
                                     width=8)
        reservoir_spin.pack(side='left')
        
        # === Run Button ===
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill='x', **pad)
        
        self.run_button = ttk.Button(
            button_frame,
            text="🚀 Run Analysis",
            command=self._run_analysis,
            style='Accent.TButton'
        )
        self.run_button.pack(pady=10)
        
        # === Progress ===
        progress_frame = ttk.LabelFrame(scrollable_frame, text="Progress", padding=10)
        progress_frame.pack(fill='x', **pad)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, length=600, mode='determinate')
        self.progress_bar.pack(fill='x', pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(anchor='w')
        
        # === Log ===
        log_frame = ttk.LabelFrame(scrollable_frame, text="Analysis Log", padding=10)
        log_frame.pack(fill='both', expand=True, **pad)
        
        self.log_text = ScrolledText(log_frame, height=15, width=100)
        self.log_text.pack(fill='both', expand=True)
    
    def _browse_input(self):
        """Browse for input file."""
        path = filedialog.askopenfilename(
            title="Select Input Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if path:
            self.input_path.set(path)
            # Auto-set output path
            if not self.output_path.get():
                self.output_path.set(os.path.dirname(path))
    
    def _browse_output(self):
        """Browse for output folder."""
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_path.set(path)
    
    def _update_cfr_label(self, *args):
        """Update CFR threshold label."""
        self.cfr_label.config(text=f"±{self.cfr_threshold.get():.2f}")
    
    def _log(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self.log_text.insert(tk.END, formatted + "\n")
        self.log_text.see(tk.END)
        self.log_messages.append(formatted)
        self.root.update()
    
    def _update_progress(self, value: float, message: str):
        """Update progress bar and label."""
        self.progress_var.set(value)
        self.progress_label.config(text=message)
        self.root.update()
    
    def _run_analysis(self):
        """Run the selected analyses."""
        # Validate inputs
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input file")
            return
        
        if not self.output_path.get():
            messagebox.showerror("Error", "Please select an output folder")
            return
        
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Input file not found")
            return
        
        # Disable button during analysis
        self.run_button.config(state='disabled')
        self.log_text.delete(1.0, tk.END)
        self.log_messages = []
        
        try:
            self._log("=" * 50)
            self._log(f"Starting {APP_TITLE}")
            self._log("=" * 50)
            
            # Load data
            self._update_progress(5, "Loading data...")
            self._log(f"Loading: {self.input_path.get()}")
            
            df = self._load_data(self.input_path.get())
            
            if df is None or len(df) == 0:
                raise ValueError("Failed to load data or data is empty")
            
            self._log(f"Loaded {len(df)} records")
            
            # Initialize results dictionary
            results = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Run selected analyses
            progress = 10
            total_analyses = sum([self.do_bathtub.get(), self.do_survival.get(),
                                 self.do_did.get(), self.do_rc.get()])
            progress_step = 80 / max(total_analyses, 1)
            
            # Bathtub Analysis
            if self.do_bathtub.get():
                self._update_progress(progress, "Running Bathtub Curve Analysis...")
                self._log("\n--- Bathtub Curve Analysis ---")
                
                bathtub_results = self._run_bathtub_analysis(df)
                if bathtub_results is not None:
                    results['Bathtub_Analysis'] = bathtub_results
                    self._log(f"Analyzed {len(bathtub_results)} equipment groups")
                
                progress += progress_step
            
            # Survival Analysis
            if self.do_survival.get():
                self._update_progress(progress, "Running Survival Analysis...")
                self._log("\n--- Survival Analysis ---")
                
                survival_results = self._run_survival_analysis(df)
                if survival_results:
                    results.update(survival_results)
                    self._log("Cox and AFT models fitted")
                
                progress += progress_step
            
            # DiD Analysis
            if self.do_did.get():
                self._update_progress(progress, "Running DiD Analysis...")
                self._log("\n--- DiD Analysis ---")
                
                did_results = self._run_did_analysis(df)
                if did_results:
                    results.update(did_results)
                    self._log("DiD analysis completed")
                
                progress += progress_step
            
            # Reservoir Computing
            if self.do_rc.get():
                self._update_progress(progress, "Running Reservoir Computing...")
                self._log("\n--- Reservoir Computing ---")
                
                rc_results = self._run_rc_analysis(df)
                if rc_results:
                    results.update(rc_results)
                    self._log("RC imputation completed")
                
                progress += progress_step
            
            # Save results
            self._update_progress(95, "Saving results...")
            output_file = self._save_results(results, timestamp)
            
            self._update_progress(100, "Complete!")
            self._log("\n" + "=" * 50)
            self._log(f"Analysis complete!")
            self._log(f"Results saved to: {output_file}")
            self._log("=" * 50)
            
            # Save log
            log_path = os.path.join(self.output_path.get(), f"analysis_log_{timestamp}.txt")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.log_messages))
            
            messagebox.showinfo("Success", f"Analysis complete!\n\nResults: {output_file}\nLog: {log_path}")
            
        except Exception as e:
            self._log(f"\nERROR: {e}")
            self._log(traceback.format_exc())
            messagebox.showerror("Error", f"Analysis failed:\n{e}")
            self._update_progress(0, "Error occurred")
        
        finally:
            self.run_button.config(state='normal')
    
    def _load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from Excel file."""
        try:
            # Try to load Survival_Data sheet first
            try:
                df = pd.read_excel(filepath, sheet_name='Survival_Data')
            except:
                df = pd.read_excel(filepath)
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            return df
        except Exception as e:
            self._log(f"Error loading data: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        for standard_name, variants in COLUMN_MAPPINGS.items():
            for variant in variants:
                if variant in df.columns:
                    df = df.rename(columns={variant: standard_name})
                    break
        return df
    
    def _run_bathtub_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run bathtub curve analysis."""
        try:
            analyzer = BathtubAnalyzer(cfr_threshold=self.cfr_threshold.get())
            
            # Check required columns
            prod_col = 'production' if 'production' in df.columns else 'cumulative_production'
            mtbf_col = 'mtbf' if 'mtbf' in df.columns else 'MTBF'
            event_col = 'event' if 'event' in df.columns else 'Event'
            eq_col = 'equipment_id' if 'equipment_id' in df.columns else 'Equipment_ID'
            
            if prod_col not in df.columns or mtbf_col not in df.columns:
                self._log("Warning: Required columns for bathtub analysis not found")
                return None
            
            result = analyzer.analyze(
                production=df[prod_col].values,
                mtbf=df[mtbf_col].values,
                event=df[event_col].values if event_col in df.columns else None,
                equipment_id='All',
                equipment_type='Bakery'
            )
            
            if result:
                self._log(f"  DFR ends at: {result.dfr_end:,.0f}" if result.dfr_end else "  DFR end: Not detected")
                self._log(f"  CFR ends at: {result.cfr_end:,.0f}" if result.cfr_end else "  CFR end: Not detected")
                self._log(f"  Weibull β: {result.overall_beta:.2f}" if not pd.isna(result.overall_beta) else "  Weibull β: N/A")
                
                return pd.DataFrame([{
                    'Equipment': result.equipment_id,
                    'DFR_End': result.dfr_end,
                    'CFR_End': result.cfr_end,
                    'Weibull_Beta': result.overall_beta,
                    'Overall_Phase': result.overall_phase,
                    'Interpretation': result.interpretation
                }])
            
            return None
            
        except Exception as e:
            self._log(f"Bathtub analysis error: {e}")
            return None
    
    def _run_survival_analysis(self, df: pd.DataFrame) -> Dict:
        """Run survival analysis."""
        results = {}
        
        try:
            analyzer = SurvivalAnalyzer()
            
            # Check required columns
            duration_col = 'mtbf' if 'mtbf' in df.columns else 'duration'
            event_col = 'event' if 'event' in df.columns else 'Event'
            
            if duration_col not in df.columns:
                self._log("Warning: Duration column not found for survival analysis")
                return results
            
            duration = df[duration_col].dropna().values
            event = df[event_col].values if event_col in df.columns else np.ones(len(duration))
            
            # Prepare covariates
            covariate_cols = []
            for col in ['treatment', 'line', 'equipment_type']:
                if col in df.columns:
                    covariate_cols.append(col)
            
            covariates = pd.get_dummies(df[covariate_cols], drop_first=True) if covariate_cols else None
            
            # Cox model
            if HAS_LIFELINES:
                cox_result = analyzer.fit_cox(duration, event, covariates)
                
                if cox_result:
                    self._log(f"  Cox Concordance Index: {cox_result.concordance_index:.3f}")
                    self._log(f"  N Events: {cox_result.n_events}/{cox_result.n_observations}")
                    
                    results['Cox_Results'] = cox_result.summary_df
                
                # AFT models
                comparison = analyzer.compare_models(duration, event, covariates)
                if len(comparison) > 0:
                    results['AFT_Comparison'] = comparison
                    best_model = comparison.iloc[0]['Distribution']
                    self._log(f"  Best AFT model: {best_model} (by AIC)")
            
            return results
            
        except Exception as e:
            self._log(f"Survival analysis error: {e}")
            return results
    
    def _run_did_analysis(self, df: pd.DataFrame) -> Dict:
        """Run DiD analysis."""
        results = {}
        
        try:
            analyzer = DiDAnalyzer(cfr_threshold=self.cfr_threshold.get())
            analyzer.df = df.copy()
            
            # Check for required columns
            if 'treatment' not in df.columns or 'post' not in df.columns:
                self._log("Warning: treatment/post columns not found, attempting to create")
                analyzer.create_did_variables()
            
            # Create treatment_post if missing
            if 'treatment_post' not in analyzer.df.columns:
                analyzer.df['treatment_post'] = analyzer.df.get('treatment', 0) * analyzer.df.get('post', 0)
            
            outcome_col = 'mtbf' if 'mtbf' in analyzer.df.columns else 'MTBF'
            
            if outcome_col not in analyzer.df.columns:
                self._log("Warning: Outcome column not found for DiD analysis")
                return results
            
            # Raw DiD
            raw_results = analyzer.calc_raw_did(outcome_col=outcome_col)
            
            if raw_results:
                did_df = pd.DataFrame([{
                    'Metric': r.metric,
                    'Group': r.group,
                    'Treated_Before': r.treated_before,
                    'Treated_After': r.treated_after,
                    'Control_Before': r.control_before,
                    'Control_After': r.control_after,
                    'DiD_Effect': r.did_effect,
                    'DiD_Effect_Pct': r.did_effect_pct,
                    'P_Value': r.p_value,
                    'Significant': r.is_significant,
                    'Interpretation': r.interpretation
                } for r in raw_results])
                
                results['DiD_Results'] = did_df
                
                for r in raw_results:
                    self._log(f"  {r.interpretation}")
            
            # TWFE
            if HAS_STATSMODELS:
                twfe_result = analyzer.run_twfe(outcome_col=outcome_col)
                
                if twfe_result:
                    results['TWFE_Results'] = pd.DataFrame([{
                        'Metric': twfe_result.metric,
                        'Coefficient': twfe_result.coefficient,
                        'Std_Error': twfe_result.std_error,
                        'P_Value': twfe_result.p_value,
                        'R_Squared': twfe_result.r_squared,
                        'Significant': twfe_result.is_significant
                    }])
                    
                    self._log(f"  TWFE coefficient: {twfe_result.coefficient:.2f} (p={twfe_result.p_value:.4f})")
            
            return results
            
        except Exception as e:
            self._log(f"DiD analysis error: {e}")
            return results
    
    def _run_rc_analysis(self, df: pd.DataFrame) -> Dict:
        """Run Reservoir Computing analysis."""
        results = {}
        
        try:
            imputer = RightCensoringImputer(
                n_reservoir=self.n_reservoir.get(),
                spectral_radius=0.95
            )
            
            # Check required columns
            mtbf_col = 'mtbf' if 'mtbf' in df.columns else 'MTBF'
            event_col = 'event' if 'event' in df.columns else 'Event'
            prod_col = 'production' if 'production' in df.columns else 'cumulative_production'
            
            if mtbf_col not in df.columns or prod_col not in df.columns:
                self._log("Warning: Required columns for RC analysis not found")
                return results
            
            mtbf = df[mtbf_col].values
            event = df[event_col].values if event_col in df.columns else np.ones(len(mtbf))
            production = df[prod_col].values
            
            # Handle NaN values
            valid_mask = ~(np.isnan(mtbf) | np.isnan(production))
            if valid_mask.sum() < 20:
                self._log("Warning: Insufficient valid data for RC analysis")
                return results
            
            result = imputer.fit_transform(
                mtbf=mtbf[valid_mask],
                event=event[valid_mask].astype(int),
                production=production[valid_mask]
            )
            
            n_imputed = result.imputation_mask.sum()
            self._log(f"  Observations imputed: {n_imputed}")
            
            if n_imputed > 0:
                avg_adjustment = (result.imputed_values - result.original_values)[result.imputation_mask].mean()
                self._log(f"  Average imputation adjustment: {avg_adjustment:.2f}")
            
            results['RC_Imputation'] = pd.DataFrame({
                'Original_MTBF': result.original_values,
                'Imputed_MTBF': result.imputed_values,
                'Is_Imputed': result.imputation_mask,
                'Confidence': result.confidence
            })
            
            return results
            
        except Exception as e:
            self._log(f"RC analysis error: {e}")
            return results
    
    def _save_results(self, results: Dict, timestamp: str) -> str:
        """Save results to Excel file."""
        output_file = os.path.join(
            self.output_path.get(),
            f"production_analysis_{timestamp}.xlsx"
        )
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Item': ['Analysis Date', 'Input File', 'CFR Threshold', 'Reservoir Nodes'],
                'Value': [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    os.path.basename(self.input_path.get()),
                    self.cfr_threshold.get(),
                    self.n_reservoir.get()
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Analysis results
            for sheet_name, df in results.items():
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    # Truncate sheet name to Excel limit
                    safe_name = sheet_name[:31]
                    df.to_excel(writer, sheet_name=safe_name, index=False)
        
        return output_file
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main entry point."""
    if HAS_TK:
        app = ProductionAnalysisGUI()
        app.run()
    else:
        print("Error: tkinter not available. Please install it or use CLI mode.")
        print("Usage: python main.py <input_file> --output <output_folder>")
        sys.exit(1)


if __name__ == '__main__':
    main()
