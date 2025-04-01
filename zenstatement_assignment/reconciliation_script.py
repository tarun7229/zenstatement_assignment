import pandas as pd
import openai
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FinancialReconciliation:
    def __init__(self):
        """Initialize paths and API key from environment variables."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)

        self.recon_data_path = os.getenv("RECON_DATA_PATH")
        self.recon_reply_path = os.getenv("RECON_REPLY_PATH")

        self.resolved_output_path = os.getenv("RESOLVED_OUTPUT_PATH")
        self.unresolved_output_path = os.getenv("UNRESOLVED_OUTPUT_PATH")
        self.unresolved_pattern_path = os.getenv("UNRESOLVED_PATTERN_PATH")
        self.auto_closure_path = os.getenv("AUTO_CLOSURE_PATH")

    def load_financial_data(self, file_path):
        """Load financial data from CSV, Excel, or JSON."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path.split('.')[-1].lower()

        if file_extension == "csv":
            return pd.read_csv(file_path, encoding="ISO-8859-1")
        elif file_extension == "xlsx":
            return pd.read_excel(file_path)
        elif file_extension == "json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def extract_amount(self, value):
        """Extract amount from JSON string in 'recon_sub_status' field."""
        try:
            parsed_json = json.loads(value)
            return parsed_json.get("amount", "")
        except (json.JSONDecodeError, TypeError):
            return ""

    def preprocess_data(self):
        """Load, process, and filter reconciliation data."""
        self.recon_data = self.load_financial_data(self.recon_data_path)
        self.recon_reply = self.load_financial_data(self.recon_reply_path)

        self.recon_data["extracted_amount"] = self.recon_data["recon_sub_status"].apply(self.extract_amount)
        not_found_sys_b = self.recon_data[self.recon_data["extracted_amount"] == "Not Found-SysB"]
        not_found_sys_b = not_found_sys_b[["txn_ref_id", "sys_a_amount_attribute_1", "sys_a_date"]]
        not_found_sys_b.columns = ["order_id", "amount", "date"]

        self.merged_data = not_found_sys_b.merge(
            self.recon_reply, left_on="order_id", right_on="Transaction ID", how="left"
        )

    def classify_comments(self):
        """Classify comments into categories using OpenAI."""
        unique_comments = self.merged_data["Comments"].dropna().unique().tolist()
        BATCH_SIZE = 20

        def classify_batch(comments):
            prompt = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(comments)])
            full_prompt = f"""
            Classify each comment into one of the categories:
            - Resolved
            - Unresolved (with reason)
            - Pending Action Required

            Comments:
            {prompt}

            Respond in a numbered list format, e.g.:
            1. Resolved
            2. Unresolved - Missing information
            3. Pending Action Required
            """
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a financial reconciliation assistant."},
                          {"role": "user", "content": full_prompt}]
            )
            return [line.split(". ", 1)[1] for line in response.choices[0].message.content.strip().split("\n") if ". " in line]

        comment_to_status = {}
        for i in range(0, len(unique_comments), BATCH_SIZE):
            batch = unique_comments[i:i + BATCH_SIZE]
            try:
                statuses = classify_batch(batch)
                comment_to_status.update(dict(zip(batch, statuses)))
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
            time.sleep(1)

        self.merged_data["Resolution Status"] = self.merged_data["Comments"].map(comment_to_status)

    def generate_unresolved_summary(self, comment):
        """Generate summary and next steps for unresolved cases."""
        prompt = f"""
        The following reconciliation case was not resolved. Provide a brief summary of why it remains unresolved, 
        along with actionable next steps for resolution.

        Comment: "{comment}"

        The output should include:
        1. A brief summary explaining why the case remains unresolved.
        2. Actionable next steps to resolve the issue.

        Format the response as:
        1. Summary: [brief explanation]
        2. Next Steps: [Provide all steps in a single sentence, separated by semicolons]

        Example Output:
        1. Summary: The case remains unresolved due to missing transaction details.
        2. Next Steps: Review transaction logs; verify payment data; contact support for error resolution; reprocess payment and confirm success.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an expert in financial reconciliation."},
                      {"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()
        lines = content.split("\n")

        # Extract the first two lines that match the expected pattern
        summary = next((line.split(": ", 1)[1] for line in lines if line.startswith("1. Summary:")), "")
        next_steps = next((line.split(": ", 1)[1] for line in lines if line.startswith("2. Next Steps:")), "")

        return summary, next_steps

    def generate_next_steps(self):
        """Generate next steps and summary for unresolved cases using multi-threading."""
        unresolved_cases = self.merged_data[self.merged_data["Resolution Status"] != "Resolved"].copy()
        with ThreadPoolExecutor(max_workers=5) as executor:
            unresolved_cases.loc[:, ["Unresolved Summary", "Next Steps"]] = list(
                executor.map(self.generate_unresolved_summary, unresolved_cases["Comments"].astype(str)))

        self.resolved_cases = self.merged_data[self.merged_data["Resolution Status"] == "Resolved"]
        self.unresolved_cases = unresolved_cases

    def export_data(self, df, output_format="csv", output_path="output.csv"):
        """Export data in CSV, JSON, or SQL for ZenStatement integration."""
        if output_format == "csv":
            df.to_csv(output_path, index=False)
        elif output_format == "json":
            df.to_json(output_path, orient="records", indent=4)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")

    def save_results(self):
        """Save processed data into CSV files for ZenStatement."""
        self.export_data(self.resolved_cases, output_format="csv", output_path=self.resolved_output_path)
        self.export_data(self.unresolved_cases, output_format="csv", output_path=self.unresolved_output_path)

        # Generate unresolved patterns
        pattern_summary = self.unresolved_cases.groupby("Comments").size().reset_index(name="count")
        self.export_data(pattern_summary, output_format="csv", output_path=self.unresolved_pattern_path)

        # Identify auto-closure patterns
        resolved_pattern_summary = self.resolved_cases.groupby("Comments").size().reset_index(name="count")
        resolved_pattern_summary["Auto_Close"] = resolved_pattern_summary["count"] >= 5
        auto_closure_cases = resolved_pattern_summary[resolved_pattern_summary["Auto_Close"]]
        self.export_data(auto_closure_cases, output_format="csv", output_path=self.auto_closure_path)

        print("Processing complete with auto-closure identification!")

    def run(self):
        """Execute the entire financial reconciliation workflow."""
        self.preprocess_data()
        self.classify_comments()
        self.generate_next_steps()
        self.save_results()

# Run the script
if __name__ == "__main__":
    reconciler = FinancialReconciliation()
    reconciler.run()
