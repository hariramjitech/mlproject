from fpdf import FPDF
import json
import os
import datetime

TITLE = "Model Evaluation Report: Cross-Validation & Random Forest"
AUTHOR = "AI Assistant"

class PDF(FPDF):
    def header(self):
        # Logo or decorative line could go here
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, TITLE, 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, text)
        self.ln()

    def add_plot(self, image_path, width=150):
        if os.path.exists(image_path):
            self.image(image_path, w=width, x=(210-width)/2)
            self.ln(5)
        else:
            self.chapter_body(f"[Image not found: {image_path}]")

def load_results(filename='evaluation_results.json'):
    with open(filename, 'r') as f:
        return json.load(f)

def create_report():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # --- Title Page info ---
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 20, "Machine Learning Model Evaluation", 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
    pdf.ln(10)

    # --- Introduction ---
    pdf.chapter_title("1. Problem Statement & Objectives")
    intro_text = (
        "This project evaluates the performance of Random Forest, Support Vector Machines (SVM), "
        "and Decision Trees using the Breast Cancer Wisconsin (Diagnostic) dataset. "
        "The primary objective is to demonstrate the effectiveness of Cross-Validation "
        "and Ensemble methods in reducing overfitting and improving model accuracy.\n\n"
        "We used Stratified K-Fold Cross-Validation (k=5) to ensure reliable performance metrics."
    )
    pdf.chapter_body(intro_text)
    
    # --- Methodology ---
    pdf.chapter_title("2. Methodology")
    method_text = (
        "Data Preprocessing:\n"
        "- Dataset: Breast Cancer Wisconsin (Diagnostic)\n"
        "- Feature Scaling: Standard Scaler applied\n"
        "Validation Technique:\n"
        "- Stratified K-Fold Cross-Validation (5 Splits)\n"
        "- Metric calculation: Accuracy, Precision, Recall, F1-Score\n"
        "Models Evaluated:\n"
        "- Decision Tree (Baseline)\n"
        "- SVM (Linear Kernel)\n"
        "- Random Forest (Base & Tuned)"
    )
    pdf.chapter_body(method_text)

    # --- Results ---
    pdf.chapter_title("3. Model Evaluation Results")
    
    results = load_results()
    
    # Create a nice format for the results
    for model_name, metrics in results.items():
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 6, model_name, 0, 1)
        pdf.set_font('Arial', '', 10)
        
        # Table-like display
        pdf.cell(45, 6, f"Accuracy: {metrics['accuracy']:.4f}", 0)
        pdf.cell(45, 6, f"F1 Score: {metrics['f1_macro']:.4f}", 0)
        pdf.ln()
        pdf.cell(45, 6, f"Precision: {metrics['precision_macro']:.4f}", 0)
        pdf.cell(45, 6, f"Recall: {metrics['recall_macro']:.4f}", 0)
        pdf.ln(8)

    pdf.chapter_body("Comparison Visualization:")
    pdf.add_plot('plots/model_accuracy_comparison.png', width=140)

    # --- Hyperparameter Tuning ---
    pdf.add_page()
    pdf.chapter_title("4. Hyperparameter Tuning (Random Forest)")
    tuning_text = (
        "We utilized GridSearchCV to optimize the Random Forest model. "
        "Parameters tuned included n_estimators, max_depth, min_samples_split, and min_samples_leaf.\n"
        "The results showed that the tuned Random Forest provided robust performance."
    )
    pdf.chapter_body(tuning_text)
    
    pdf.chapter_body("Confusion Matrices for Tuned Random Forest vs SVM:")
    pdf.add_plot('plots/rf_tuned_cm.png', width=120)
    pdf.ln(2)
    pdf.add_plot('plots/svm_cm.png', width=120)

    # --- Conclusion ---
    pdf.chapter_title("5. Conclusion")
    conclusion_text = (
        "The Stratified K-Fold cross-validation results demonstrate the generalization capability of the models. "
        "Random Forest, particularly after tuning, typically yields high accuracy and F1-scores, "
        "handling the feature space effectively compared to a single Decision Tree. "
        "SVM also performed competitively on this dataset."
    )
    pdf.chapter_body(conclusion_text)

    output_path = "Model_Evaluation_Report.pdf"
    pdf.output(output_path)
    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    create_report()
