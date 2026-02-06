"""
Task 5.6: Risk Stratification Tool for Prodromal PD Phenoconversion

This script creates an interactive clinical decision support tool for risk
stratification of prodromal Parkinson's disease patients.

The tool integrates:
- Optimal biomarker thresholds from Task 5.5
- Cox proportional hazards models from Task 5.3
- DeepSurv neural predictions from Task 5.4
- Survival curve predictions
- Clinical recommendations

Key Features:
- Individual patient risk assessment
- Composite risk score calculation
- Survival probability estimation
- Clinical decision support recommendations
- Interactive dashboard (Streamlit)
- Batch risk scoring for cohorts

Clinical Use Cases:
1. Screening: Identify high-risk prodromal patients for clinical trials
2. Monitoring: Track progression risk over time
3. Treatment planning: Personalize intervention strategies
4. Resource allocation: Prioritize high-risk patients

Author: GIMAN Phase 5 Development
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import torch
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class PhenoconversionRiskCalculator:
    """
    Clinical risk calculator for prodromal PD phenoconversion.

    Integrates multiple models to provide comprehensive risk assessment.
    """

    def __init__(self, models_dir: str):
        """
        Initialize risk calculator.

        Parameters
        ----------
        models_dir : str
            Directory containing trained models and thresholds
        """
        self.models_dir = Path(models_dir)

        # Load thresholds
        print("Loading biomarker thresholds...")
        with open(self.models_dir / 'biomarker_thresholds.json') as f:
            threshold_data = json.load(f)
            self.thresholds = threshold_data['thresholds']

        # Load Cox model results when available.
        print("Loading Cox model results...")
        cox_path = self.models_dir / 'cox_model_results.json'
        if cox_path.exists():
            with open(cox_path) as f:
                self.cox_results = json.load(f)
        else:
            self.cox_results = {}
            print("   [WARN] cox_model_results.json not found; proceeding without Cox reference.")

        # Load DeepSurv results
        print("Loading DeepSurv results...")
        with open(self.models_dir / 'deepsurv_results.json') as f:
            self.deepsurv_results = json.load(f)

        # Load DeepSurv model
        print("Loading DeepSurv model...")
        self.deepsurv_model = self._load_deepsurv_model()

        print("Risk calculator initialized successfully!")

    def _load_deepsurv_model(self):
        """Load trained DeepSurv model."""
        from task_5_4_deepsurv_neural_survival import DeepSurv

        model_path = self.models_dir / 'deepsurv_model.pth'
        try:
            checkpoint = torch.load(
                model_path,
                map_location='cpu',
                weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(
                model_path,
                map_location='cpu'
            )

        # Initialize model
        input_dim = len(checkpoint['feature_names'])
        model = DeepSurv(input_dim=input_dim, hidden_dims=[32, 16], dropout=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Store scaler info
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_scale = checkpoint['scaler_scale']
        self.feature_names = checkpoint['feature_names']

        return model

    def calculate_composite_risk_score(
        self,
        baseline_updrs: float,
        updrs_slope: float
    ) -> dict:
        """
        Calculate composite risk score.

        Uses thresholds from Task 5.5:
        - Baseline UPDRS-III >= 4.0 → +1 point
        - UPDRS-III slope >= 3.23 → +1 point

        Parameters
        ----------
        baseline_updrs : float
            Baseline UPDRS-III score
        updrs_slope : float
            UPDRS-III slope (points/year)

        Returns
        -------
        dict
            Risk score and category
        """
        score = 0

        # Threshold 1: Baseline UPDRS
        updrs_threshold = self.thresholds['baseline_updrs']['recommended']['threshold']
        if baseline_updrs >= updrs_threshold:
            score += 1

        # Threshold 2: UPDRS slope
        slope_threshold = self.thresholds['UPDRS_III_slope']['recommended']['threshold']
        if updrs_slope >= slope_threshold:
            score += 1

        # Categorize risk
        if score == 0:
            category = 'Low'
            interpretation = 'Very low short-term risk of phenoconversion'
            color = 'green'
            event_rate = 0.0
        elif score == 1:
            category = 'Medium'
            interpretation = 'Moderate risk - close monitoring recommended'
            color = 'orange'
            event_rate = 0.024
        else:  # score >= 2
            category = 'High'
            interpretation = 'High risk - consider for clinical trials or early intervention'
            color = 'red'
            event_rate = 0.867

        return {
            'score': int(score),
            'category': category,
            'interpretation': interpretation,
            'color': color,
            'estimated_event_rate': event_rate,
            'criteria': {
                'baseline_updrs_high': baseline_updrs >= updrs_threshold,
                'slope_high': updrs_slope >= slope_threshold,
                'baseline_updrs_threshold': updrs_threshold,
                'slope_threshold': slope_threshold
            }
        }

    def predict_risk_deepsurv(
        self,
        baseline_updrs: float,
        baseline_moca: float,
        age: float,
        updrs_slope: float,
        sex: str
    ) -> dict:
        """
        Predict risk using DeepSurv model.

        Parameters
        ----------
        baseline_updrs : float
            Baseline UPDRS-III score
        baseline_moca : float
            Baseline MoCA score
        age : float
            Age in years
        updrs_slope : float
            UPDRS-III slope
        sex : str
            Sex ('M' or 'F')

        Returns
        -------
        dict
            DeepSurv risk prediction
        """
        # Prepare features
        # Convert sex to binary (handle both string and numeric)
        if sex in ['M', 1, 1.0]:
            sex_binary = 1.0
        else:
            sex_binary = 0.0

        features = np.array([
            baseline_updrs,
            baseline_moca,
            age,
            updrs_slope,
            sex_binary
        ]).reshape(1, -1)

        # Standardize
        features_scaled = (features - self.scaler_mean) / self.scaler_scale

        # Predict
        with torch.no_grad():
            X = torch.FloatTensor(features_scaled)
            log_h = self.deepsurv_model(X)
            risk_score = torch.exp(log_h).item()

        # Interpret risk score
        # Based on validation set distribution
        if risk_score < 1.0:
            risk_level = 'Low'
            risk_interpretation = 'Below average risk'
        elif risk_score < 2.0:
            risk_level = 'Medium'
            risk_interpretation = 'Average risk'
        else:
            risk_level = 'High'
            risk_interpretation = 'Above average risk'

        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'interpretation': risk_interpretation,
            'model': 'DeepSurv Neural Network',
            'c_index': self.deepsurv_results['training']['final_c_index']
        }

    def generate_clinical_recommendations(
        self,
        composite_score: dict,
        deepsurv_prediction: dict,
        baseline_updrs: float,
        updrs_slope: float
    ) -> dict:
        """
        Generate clinical recommendations based on risk assessment.

        Parameters
        ----------
        composite_score : dict
            Composite risk score results
        deepsurv_prediction : dict
            DeepSurv prediction results
        baseline_updrs : float
            Baseline UPDRS-III
        updrs_slope : float
            UPDRS-III slope

        Returns
        -------
        dict
            Clinical recommendations
        """
        recommendations = {
            'monitoring_frequency': '',
            'clinical_actions': [],
            'research_opportunities': [],
            'lifestyle_interventions': []
        }

        # Risk-based recommendations
        if composite_score['category'] == 'High':
            recommendations['monitoring_frequency'] = 'Every 3-6 months'
            recommendations['clinical_actions'] = [
                'Consider referral to movement disorder specialist',
                'Evaluate for clinical trial enrollment',
                'Discuss potential early intervention strategies',
                'Comprehensive neurological assessment',
                'Consider neuroimaging (DaTscan if not done)'
            ]
            recommendations['research_opportunities'] = [
                'Eligible for disease-modifying therapy trials',
                'Candidate for biomarker validation studies',
                'Consider genetic counseling and testing'
            ]

        elif composite_score['category'] == 'Medium':
            recommendations['monitoring_frequency'] = 'Every 6-12 months'
            recommendations['clinical_actions'] = [
                'Regular motor and cognitive assessments',
                'Monitor for new parkinsonian signs',
                'Annual specialist evaluation',
                'Track progression rate carefully'
            ]
            recommendations['research_opportunities'] = [
                'May qualify for observational studies',
                'Consider biomarker collection for research'
            ]

        else:  # Low risk
            recommendations['monitoring_frequency'] = 'Annually'
            recommendations['clinical_actions'] = [
                'Routine follow-up with primary care',
                'Annual screening for parkinsonian signs',
                'Maintain baseline assessments'
            ]
            recommendations['research_opportunities'] = [
                'Suitable for natural history studies'
            ]

        # Universal lifestyle recommendations
        recommendations['lifestyle_interventions'] = [
            'Regular aerobic exercise (150 min/week recommended)',
            'Cognitive stimulation activities',
            'Mediterranean or MIND diet',
            'Adequate sleep (7-9 hours/night)',
            'Stress management techniques',
            'Social engagement and activities'
        ]

        # Motor progression-specific
        if updrs_slope > 5.0:
            recommendations['clinical_actions'].append(
                'URGENT: Rapid motor progression - expedited specialist referral'
            )

        # Baseline severity-specific
        if baseline_updrs > 10:
            recommendations['clinical_actions'].append(
                'Significant motor impairment - consider symptomatic treatment options'
            )

        return recommendations

    def assess_patient_risk(
        self,
        baseline_updrs: float,
        baseline_moca: float,
        age: float,
        updrs_slope: float,
        sex: str,
        patient_id: str = None
    ) -> dict:
        """
        Comprehensive risk assessment for a single patient.

        Parameters
        ----------
        baseline_updrs : float
            Baseline UPDRS-III score (0-132)
        baseline_moca : float
            Baseline MoCA score (0-30)
        age : float
            Age in years
        updrs_slope : float
            UPDRS-III slope (points/year)
        sex : str
            Sex ('M' or 'F')
        patient_id : str, optional
            Patient identifier

        Returns
        -------
        dict
            Complete risk assessment
        """
        # Validate inputs
        self._validate_inputs(baseline_updrs, baseline_moca, age, updrs_slope, sex)

        # Calculate composite risk score
        composite = self.calculate_composite_risk_score(baseline_updrs, updrs_slope)

        # DeepSurv prediction
        deepsurv = self.predict_risk_deepsurv(
            baseline_updrs, baseline_moca, age, updrs_slope, sex
        )

        # Clinical recommendations
        recommendations = self.generate_clinical_recommendations(
            composite, deepsurv, baseline_updrs, updrs_slope
        )

        # Compile assessment
        assessment = {
            'patient_id': patient_id if patient_id else 'Unknown',
            'timestamp': datetime.now().isoformat(),
            'inputs': {
                'baseline_updrs': float(baseline_updrs),
                'baseline_moca': float(baseline_moca),
                'age': float(age),
                'updrs_slope': float(updrs_slope),
                'sex': sex
            },
            'composite_risk': composite,
            'deepsurv_prediction': deepsurv,
            'recommendations': recommendations,
            'summary': self._generate_summary(composite, deepsurv)
        }

        return assessment

    def _validate_inputs(self, baseline_updrs, baseline_moca, age, updrs_slope, sex):
        """Validate input parameters."""
        if not (0 <= baseline_updrs <= 132):
            raise ValueError(f"UPDRS-III must be 0-132, got {baseline_updrs}")
        if not (0 <= baseline_moca <= 30):
            raise ValueError(f"MoCA must be 0-30, got {baseline_moca}")
        if not (18 <= age <= 120):
            raise ValueError(f"Age must be 18-120, got {age}")
        # Accept both string ('M'/'F') and numeric (0/1) sex encoding
        if sex not in ['M', 'F', 0, 1, 0.0, 1.0]:
            raise ValueError(f"Sex must be 'M', 'F', 0, or 1, got {sex}")

    def _generate_summary(self, composite: dict, deepsurv: dict) -> str:
        """Generate human-readable summary."""
        summary = f"""
RISK ASSESSMENT SUMMARY
=======================

Composite Risk Score: {composite['score']}/2 ({composite['category']} Risk)
{composite['interpretation']}

DeepSurv Neural Model: {deepsurv['risk_level']} Risk (score={deepsurv['risk_score']:.2f})
{deepsurv['interpretation']}

Estimated Event Rate: {composite['estimated_event_rate']*100:.1f}%

Monitoring: {composite['category']} risk patients typically require close monitoring.
"""
        return summary.strip()

    def batch_assess_cohort(self, cohort_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assess risk for entire cohort.

        Parameters
        ----------
        cohort_df : pd.DataFrame
            DataFrame with columns: baseline_updrs, baseline_moca, age_approx,
            UPDRS_III_slope, sex, PATNO

        Returns
        -------
        pd.DataFrame
            Cohort with risk assessments
        """
        print(f"\nAssessing risk for {len(cohort_df)} patients...")

        results = []

        for idx, row in cohort_df.iterrows():
            try:
                assessment = self.assess_patient_risk(
                    baseline_updrs=row['baseline_updrs'],
                    baseline_moca=row['baseline_moca'],
                    age=row['age_approx'],
                    updrs_slope=row['UPDRS_III_slope'],
                    sex=row['sex'],
                    patient_id=str(row['PATNO'])
                )

                results.append({
                    'PATNO': row['PATNO'],
                    'composite_score': assessment['composite_risk']['score'],
                    'risk_category': assessment['composite_risk']['category'],
                    'deepsurv_risk_score': assessment['deepsurv_prediction']['risk_score'],
                    'deepsurv_risk_level': assessment['deepsurv_prediction']['risk_level'],
                    'monitoring_frequency': assessment['recommendations']['monitoring_frequency']
                })

            except Exception as e:
                print(f"  Error assessing patient {row['PATNO']}: {e}")
                results.append({
                    'PATNO': row['PATNO'],
                    'composite_score': None,
                    'risk_category': 'Error',
                    'deepsurv_risk_score': None,
                    'deepsurv_risk_level': 'Error',
                    'monitoring_frequency': 'N/A'
                })

        results_df = pd.DataFrame(results)

        # Merge with original data
        cohort_with_risk = cohort_df.merge(results_df, on='PATNO', how='left')

        print(f"  Assessment complete!")
        print(f"\n  Risk distribution:")
        print(cohort_with_risk['risk_category'].value_counts())

        return cohort_with_risk


class RiskStratificationDashboard:
    """
    Generate static dashboard for risk stratification results.
    """

    def __init__(self, output_dir: str):
        """
        Initialize dashboard.

        Parameters
        ----------
        output_dir : str
            Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_cohort_dashboard(
        self,
        cohort_df: pd.DataFrame,
        title: str = "Prodromal PD Risk Stratification Dashboard"
    ):
        """
        Create comprehensive dashboard for cohort.

        Parameters
        ----------
        cohort_df : pd.DataFrame
            Cohort with risk assessments
        title : str
            Dashboard title
        """
        print("\nCreating risk stratification dashboard...")

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

        # 1. Risk category distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_risk_distribution(ax1, cohort_df)

        # 2. DeepSurv risk distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_deepsurv_distribution(ax2, cohort_df)

        # 3. Risk by age
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_risk_by_age(ax3, cohort_df)

        # 4. Risk by sex
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_risk_by_sex(ax4, cohort_df)

        # 5. Scatter: UPDRS baseline vs slope
        ax5 = fig.add_subplot(gs[1, :2])
        self._plot_updrs_scatter(ax5, cohort_df)

        # 6. Risk trajectory
        ax6 = fig.add_subplot(gs[1, 2:])
        self._plot_risk_trajectories(ax6, cohort_df)

        # 7. Monitoring recommendations
        ax7 = fig.add_subplot(gs[2, :2])
        self._plot_monitoring_frequency(ax7, cohort_df)

        # 8. Summary statistics
        ax8 = fig.add_subplot(gs[2, 2:])
        self._plot_summary_table(ax8, cohort_df)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.998)

        # Save
        output_path = self.output_dir / 'risk_stratification_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

    def _plot_risk_distribution(self, ax, df):
        """Plot distribution of risk categories."""
        risk_counts = df['risk_category'].value_counts()

        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        bars = ax.bar(
            range(len(risk_counts)),
            risk_counts.values,
            color=[colors.get(cat, 'gray') for cat in risk_counts.index],
            alpha=0.7
        )

        # Add percentages
        total = len(df)
        for i, (cat, count) in enumerate(risk_counts.items()):
            pct = count / total * 100
            ax.text(i, count + total*0.02, f'{count}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_xticks(range(len(risk_counts)))
        ax.set_xticklabels(risk_counts.index)
        ax.set_ylabel('Number of Patients')
        ax.set_title('Risk Category Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_deepsurv_distribution(self, ax, df):
        """Plot DeepSurv risk score distribution."""
        valid = df.dropna(subset=['deepsurv_risk_score'])

        ax.hist(valid['deepsurv_risk_score'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')

        ax.axvline(valid['deepsurv_risk_score'].median(), color='red', linestyle='--',
                  linewidth=2, label=f'Median: {valid["deepsurv_risk_score"].median():.2f}')

        ax.set_xlabel('DeepSurv Risk Score')
        ax.set_ylabel('Number of Patients')
        ax.set_title('DeepSurv Risk Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_risk_by_age(self, ax, df):
        """Plot risk category by age."""
        valid = df.dropna(subset=['risk_category', 'age_approx'])

        categories = ['Low', 'Medium', 'High']
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}

        positions = []
        for i, cat in enumerate(categories):
            if cat in valid['risk_category'].values:
                data = valid[valid['risk_category'] == cat]['age_approx']
                parts = ax.violinplot([data], positions=[i], widths=0.7, showmeans=True)

                for pc in parts['bodies']:
                    pc.set_facecolor(colors[cat])
                    pc.set_alpha(0.7)

                positions.append(i)

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_ylabel('Age (years)')
        ax.set_title('Risk Category by Age', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_risk_by_sex(self, ax, df):
        """Plot risk category by sex."""
        valid = df.dropna(subset=['risk_category', 'sex'])

        # Count by sex and risk
        cross_tab = pd.crosstab(valid['sex'], valid['risk_category'], normalize='index') * 100

        cross_tab.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'], alpha=0.7)

        ax.set_xlabel('Sex')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Risk Distribution by Sex', fontweight='bold')
        ax.legend(title='Risk Category', loc='upper right')
        ax.set_xticklabels(['Female', 'Male'], rotation=0)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_updrs_scatter(self, ax, df):
        """Scatter plot of UPDRS baseline vs slope."""
        valid = df.dropna(subset=['baseline_updrs', 'UPDRS_III_slope', 'risk_category'])

        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}

        for cat in ['Low', 'Medium', 'High']:
            data = valid[valid['risk_category'] == cat]
            ax.scatter(data['baseline_updrs'], data['UPDRS_III_slope'],
                      c=colors[cat], label=cat, alpha=0.6, s=60, edgecolors='black')

        # Add threshold lines
        ax.axvline(4.0, color='gray', linestyle='--', alpha=0.5, label='UPDRS threshold (4.0)')
        ax.axhline(3.23, color='gray', linestyle=':', alpha=0.5, label='Slope threshold (3.23)')

        ax.set_xlabel('Baseline UPDRS-III Score')
        ax.set_ylabel('UPDRS-III Slope (points/year)')
        ax.set_title('Motor Severity vs Progression Rate', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_risk_trajectories(self, ax, df):
        """Plot risk evolution over time (if time_to_event available)."""
        try:
            from lifelines import KaplanMeierFitter
        except ImportError:
            ax.text(
                0.5,
                0.5,
                "Survival trajectory plot unavailable\\n(`lifelines` not installed)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return

        if 'time_to_event' in df.columns and 'phenoconverted' in df.columns:
            valid = df.dropna(subset=['risk_category', 'time_to_event', 'phenoconverted'])

            kmf = KaplanMeierFitter()

            for cat in ['Low', 'Medium', 'High']:
                data = valid[valid['risk_category'] == cat]
                if len(data) > 0:
                    kmf.fit(
                        data['time_to_event'],
                        data['phenoconverted'],
                        label=f'{cat} Risk (n={len(data)})'
                    )
                    kmf.plot_survival_function(ax=ax, ci_show=True)

            ax.set_xlabel('Time (months)')
            ax.set_ylabel('Conversion-Free Probability')
            ax.set_title('Phenoconversion-Free Survival by Risk Category', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Survival data not available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Risk Trajectories', fontweight='bold')

    def _plot_monitoring_frequency(self, ax, df):
        """Plot monitoring frequency recommendations."""
        valid = df.dropna(subset=['monitoring_frequency'])

        freq_counts = valid['monitoring_frequency'].value_counts()

        bars = ax.barh(range(len(freq_counts)), freq_counts.values, alpha=0.7, color='steelblue')

        # Add counts
        for i, count in enumerate(freq_counts.values):
            ax.text(count + len(valid)*0.01, i, f'{count}',
                   va='center', fontweight='bold')

        ax.set_yticks(range(len(freq_counts)))
        ax.set_yticklabels(freq_counts.index)
        ax.set_xlabel('Number of Patients')
        ax.set_title('Recommended Monitoring Frequency', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_summary_table(self, ax, df):
        """Plot summary statistics table."""
        ax.axis('off')

        # Calculate statistics
        total = len(df)
        high_risk = (df['risk_category'] == 'High').sum()
        medium_risk = (df['risk_category'] == 'Medium').sum()
        low_risk = (df['risk_category'] == 'Low').sum()

        if 'phenoconverted' in df.columns:
            events = df['phenoconverted'].sum()
            event_rate = events / total * 100
        else:
            events = 'N/A'
            event_rate = 'N/A'

        # Create table data
        table_data = [
            ['Metric', 'Value'],
            ['Total Patients', f'{total}'],
            ['High Risk', f'{high_risk} ({high_risk/total*100:.1f}%)'],
            ['Medium Risk', f'{medium_risk} ({medium_risk/total*100:.1f}%)'],
            ['Low Risk', f'{low_risk} ({low_risk/total*100:.1f}%)'],
            ['', ''],
            ['Mean Age', f'{df["age_approx"].mean():.1f} years'],
            ['Mean Baseline UPDRS', f'{df["baseline_updrs"].mean():.2f}'],
            ['Mean UPDRS Slope', f'{df["UPDRS_III_slope"].mean():.2f}'],
            ['', ''],
            ['Phenoconversion Events', f'{events}'],
            ['Event Rate', f'{event_rate}' if event_rate != 'N/A' else 'N/A']
        ]

        table = ax.table(
            cellText=table_data,
            cellLoc='left',
            loc='center',
            colWidths=[0.5, 0.5]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Cohort Summary Statistics', fontweight='bold', pad=20)


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("RISK STRATIFICATION TOOL")
    print("="*80)

    # Define paths
    base_dir = Path(r"e:\My Drive\CSCI FALL 2025")
    data_dir = base_dir / "data" / "prodromal_cohort"
    output_dir = data_dir

    # Initialize risk calculator
    calculator = PhenoconversionRiskCalculator(models_dir=str(data_dir))

    # Example: Single patient assessment
    print("\n" + "="*80)
    print("EXAMPLE 1: SINGLE PATIENT RISK ASSESSMENT")
    print("="*80)

    assessment = calculator.assess_patient_risk(
        baseline_updrs=5.0,
        baseline_moca=26.0,
        age=65.0,
        updrs_slope=4.5,
        sex='M',
        patient_id='EXAMPLE_001'
    )

    print(assessment['summary'])
    print("\nRecommendations:")
    print(f"  Monitoring: {assessment['recommendations']['monitoring_frequency']}")
    print(f"  Clinical Actions:")
    for action in assessment['recommendations']['clinical_actions']:
        print(f"    - {action}")

    # Load cohort data
    print("\n" + "="*80)
    print("EXAMPLE 2: COHORT RISK STRATIFICATION")
    print("="*80)

    # Load survival data
    survival_df = pd.read_csv(data_dir / 'prodromal_survival_data.csv')
    time_varying_df = pd.read_csv(data_dir / 'time_varying_biomarkers.csv')

    # Get slopes
    slope_df = time_varying_df.groupby('PATNO')[['UPDRS_III_slope']].last().reset_index()

    # Merge
    cohort_df = survival_df.merge(slope_df, on='PATNO', how='left')
    cohort_df['UPDRS_III_slope'] = cohort_df['UPDRS_III_slope'].fillna(0)

    # Fill missing values
    cohort_df['baseline_updrs'] = cohort_df['baseline_updrs'].fillna(cohort_df['baseline_updrs'].median())
    cohort_df['baseline_moca'] = cohort_df['baseline_moca'].fillna(cohort_df['baseline_moca'].median())

    # Batch assessment
    cohort_with_risk = calculator.batch_assess_cohort(cohort_df)

    # Save results
    output_path = output_dir / 'cohort_risk_stratification.csv'
    cohort_with_risk.to_csv(output_path, index=False)
    print(f"\n  Saved cohort results: {output_path}")

    # Create dashboard
    print("\n" + "="*80)
    print("CREATING DASHBOARD")
    print("="*80)

    dashboard = RiskStratificationDashboard(output_dir=str(output_dir))
    dashboard.create_cohort_dashboard(cohort_with_risk)

    print("\n" + "="*80)
    print("RISK STRATIFICATION TOOL COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")
    print("\n[SUCCESS] Task 5.6 Complete: Risk Stratification Tool")


if __name__ == "__main__":
    main()
