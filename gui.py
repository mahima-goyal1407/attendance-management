# ---------------- GUI ----------------
root = tk.Tk()
root.title("Face Attendance System")
root.geometry("400x350")

tk.Label(root, text="Login", font=("Arial", 16)).pack(pady=10)

tk.Label(root, text="Roll Number").pack()
entry_user = tk.Entry(root)
entry_user.pack()

tk.Label(root, text="Password").pack()
entry_pass = tk.Entry(root, show="*")
entry_pass.pack()

tk.Button(root, text="Login", command=login).pack(pady=10)

tk.Label(root, text="Enter Subject Name").pack()
entry_subject = tk.Entry(root)
entry_subject.pack()

tk.Button(root, text="Start Attendance", command=start_recognition).pack(pady=20)

tk.Button(root, text="Exit", command=root.quit).pack()

root.mainloop()
