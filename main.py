import os
from PIL import ImageTk
from app_controller import MainApp

if __name__ == '__main__':
    app = MainApp()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ico_path = os.path.join(script_dir, "logo.ico")
    try:
        app.iconbitmap('logo.ico')
    except Exception as e:
        icon = ImageTk.PhotoImage(file=ico_path)
        app.wm_iconphoto(True, icon)
    app.mainloop()

