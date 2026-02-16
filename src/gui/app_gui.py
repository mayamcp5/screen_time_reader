import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from src.ios.overall import process_ios_overall_screenshot
from src.ios.activity import process_ios_category_screenshot
from src.android.overall import process_android_overall_screenshot
from src.android.activity_history import process_android_activity_history

CAROLINA_BLUE = "#4B9CD3"
SOCIAL_COLOR = "#4B9CD3"
ENTERTAINMENT_COLOR = "#F39C12"
OVERALL_COLOR = "#7F8C8D"

class ScreenTimeApp:
    def __init__(self, root):
        self.root = root
        root.title("Screen Time Extractor")
        root.geometry("900x700")
        root.configure(bg="white")

        # Platform selection
        tk.Label(root, text="Select Platform:", bg="white", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.platform_var = tk.StringVar(value="iOS")
        tk.OptionMenu(root, self.platform_var, "iOS", "Android", command=self.toggle_platform).pack()

        # Frame for file uploads
        self.upload_frame = tk.Frame(root, bg="white")
        self.upload_frame.pack(pady=10)

        # Overall Image upload
        tk.Label(self.upload_frame, text="Overall Screenshot(s):", bg="white").grid(row=0, column=0, sticky="w")
        self.overall_btn = tk.Button(self.upload_frame, text="Upload", command=self.upload_overall)
        self.overall_btn.grid(row=0, column=1, padx=5)

        # Category/Activity upload
        tk.Label(self.upload_frame, text="Category/Activity Screenshot(s):", bg="white").grid(row=1, column=0, sticky="w")
        self.category_btn = tk.Button(self.upload_frame, text="Upload", command=self.upload_category)
        self.category_btn.grid(row=1, column=1, padx=5)

        # Process and reset buttons
        self.process_btn = tk.Button(root, text="Process Screenshots", command=self.process_files, state=tk.DISABLED)
        self.process_btn.pack(pady=5)
        self.clear_btn = tk.Button(root, text="Clear / Reset", command=self.clear_all)
        self.clear_btn.pack(pady=5)

        # Notebook for results
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both", pady=10)

        # Export CSV
        self.export_btn = tk.Button(root, text="Export CSV", command=self.export_csv, state=tk.DISABLED)
        self.export_btn.pack(pady=5)

        self.overall_files = []
        self.category_files = []
        self.results = []

    def toggle_platform(self, val):
        self.clear_all()

    def upload_overall(self):
        files = filedialog.askopenfilenames(filetypes=[("Image files","*.png *.jpg *.jpeg")])
        if files:
            self.overall_files = files
            self.update_process_state()

    def upload_category(self):
        files = filedialog.askopenfilenames(filetypes=[("Image files","*.png *.jpg *.jpeg")])
        if files:
            self.category_files = files
            self.update_process_state()

    def update_process_state(self):
        if self.overall_files or self.category_files:
            self.process_btn.config(state=tk.NORMAL)
        else:
            self.process_btn.config(state=tk.DISABLED)

    def process_files(self):
        if not (self.overall_files or self.category_files):
            messagebox.showerror("Error", "No files selected!")
            return

        self.results = []
        platform = self.platform_var.get()

        try:
            # iOS
            if platform == "iOS":
                if self.overall_files:
                    for f in self.overall_files:
                        res = process_ios_overall_screenshot(f)
                        self.results.append(res)
                        self.display_ios_overall(res, f"Overall: {f}")
                if self.category_files:
                    for f in self.category_files:
                        res = process_ios_category_screenshot(f)
                        self.results.append(res)
                        self.display_ios_category(res, f"Category: {f}")

            # Android
            elif platform == "Android":
                if self.overall_files:
                    for f in self.overall_files:
                        res = process_android_overall_screenshot(f)
                        self.results.append(res)
                        self.display_android_overall(res, f"Overall: {f}")
                if self.category_files:
                    res = process_android_activity_history(self.category_files)
                    self.results.append(res)
                    self.display_android_activity(res, "Activity History")

            self.export_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    # ----- Display Functions -----
    def display_ios_overall(self, data, title):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)

        # Summary
        tk.Label(frame, text=f"Date: {data.get('date','N/A')} | Total Time: {data.get('total_time','0h 0m')}", font=("Helvetica", 12, "bold")).pack(pady=5)

        # Top Categories Table
        self.create_table(frame, ["Category", "Time"], [[c['name'], c['time']] for c in data.get('categories', [])], "Top Categories")

        # Top Apps Table
        self.create_table(frame, ["App", "Time"], [[a['name'], a['time']] for a in data.get('top_apps', [])], "Top Apps")

        # Hourly Usage Table
        hourly = data.get("hourly_usage", {})
        rows = [[h, d.get("overall",0), d.get("social",0), d.get("entertainment",0)] for h,d in hourly.items() if h != "ymax_pixels"]
        self.create_table(frame, ["Hour", "Overall", "Social", "Entertainment"], rows, "Hourly Usage", color_cols={"Overall": OVERALL_COLOR, "Social": SOCIAL_COLOR, "Entertainment": ENTERTAINMENT_COLOR})

    def display_ios_category(self, data, title):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        tk.Label(frame, text=f"Category: {data.get('category','N/A')} | Total Time: {data.get('total_time','0h 0m')}", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.create_table(frame, ["App", "Time"], [[a['name'], a['time']] for a in data.get('apps',[])], f"{data.get('category','Category')} Apps")

    def display_android_overall(self, data, title):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        tk.Label(frame, text=f"Date: {data.get('date','N/A')} | Total Time: {data.get('total_time','0h 0m')}", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.create_table(frame, ["App", "Time"], [[a['name'], a['time']] for a in data.get('top_apps',[])], "Top Apps")

    def display_android_activity(self, data, title):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        self.create_table(frame, ["App", "Time"], [[a['name'], a['time']] for a in data.get('apps',[])], "Activity Apps")

    # Generic table creator
    def create_table(self, parent, headers, rows, title, color_cols=None):
        tk.Label(parent, text=title, font=("Helvetica", 11, "bold")).pack(pady=3)
        tree = ttk.Treeview(parent, columns=headers, show="headings", height=min(len(rows)+1, 10))
        tree.pack(padx=5, pady=2, fill="x")
        for h in headers:
            tree.heading(h, text=h)
            tree.column(h, anchor="center")
        if color_cols:
            for i, row in enumerate(rows):
                tags = []
                for idx, h in enumerate(headers):
                    if h in color_cols:
                        tags.append(h)
                tree.insert("", "end", values=row, tags=tags)
            for col, color in color_cols.items():
                tree.tag_configure(col, background=color, foreground="white")
        else:
            for row in rows:
                tree.insert("", "end", values=row)

    # ----- Clear / Reset -----
    def clear_all(self):
        self.overall_files = []
        self.category_files = []
        self.results = []
        self.process_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        # Remove all tabs safely
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)


    # ----- Export CSV -----
    def export_csv(self):
        if not self.results:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not path:
            return
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for r in self.results:
                if "hourly_usage" in r:
                    writer.writerow(["Hourly Usage"])
                    writer.writerow(["Hour","Overall","Social","Entertainment"])
                    for h,d in r["hourly_usage"].items():
                        if h != "ymax_pixels":
                            writer.writerow([h,d.get("overall",0),d.get("social",0),d.get("entertainment",0)])
                if "categories" in r:
                    writer.writerow(["Top Categories"])
                    writer.writerow(["Category","Time"])
                    for c in r["categories"]:
                        writer.writerow([c['name'],c['time']])
                if "top_apps" in r:
                    writer.writerow(["Top Apps"])
                    writer.writerow(["App","Time"])
                    for a in r["top_apps"]:
                        writer.writerow([a['name'],a['time']])
                if "apps" in r:
                    writer.writerow(["Activity / Category Apps"])
                    writer.writerow(["App","Time"])
                    for a in r['apps']:
                        writer.writerow([a['name'],a['time']])
        messagebox.showinfo("Success", f"CSV exported to {path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenTimeApp(root)
    root.mainloop()
