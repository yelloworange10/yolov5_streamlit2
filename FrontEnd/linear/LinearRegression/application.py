import tkinter as tk


# 创建主窗口
root = tk.Tk()
root.title("慧眼识桃")
root.geometry('428x926')#iphone12 pro max

# 创建标签
label = tk.Label(root, text="欢迎使用Python GUI", font=("Helvetica", 16))
label.pack(pady=20)

# 创建按钮点击事件处理函数
def button_click():
    label.config(text="按钮被点击了！")

# 创建按钮
button = tk.Button(root, text="点击我", command=button_click)
button.pack()

# 运行主循环
root.mainloop()
