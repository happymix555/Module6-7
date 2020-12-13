import tkinter as tk
import tkinter.ttk as ttk
from PC2PIC2_2 import *

lnwmodule = serial.Serial()
sleep(2)
lnwmodule.baudrate = 19200
lnwmodule.port = 'COM12'
lnwmodule.rts = 0
lnwmodule.open()

class LnwApp:
    def __init__(self, master=None):
        # build ui
        master.title("LNWmodule GUI")
        self.frame_1 = tk.Frame(master)
        self.frame_2 = tk.Frame(self.frame_1)
        self.label_1 = tk.Label(self.frame_2)
        self.logolek_png = tk.PhotoImage(file='logolek.png')
        self.label_1.config(background='#ffff80', image=self.logolek_png)
        self.label_1.pack(padx='5', pady='5', side='top')
        self.button_1 = tk.Button(self.frame_2)
        self.button_1.config(font='{Impact} 16 {}', justify='left', text='START')
        self.button_1.pack(padx='5', pady='5', side='top')
        self.button_2 = tk.Button(self.frame_2)
        self.button_2.config(font='{Impact} 16 {}', justify='left', takefocus=False, text='Capture template')
        self.button_2.pack(padx='5', pady='5', side='top')
        self.button_3 = tk.Button(self.frame_2)
        self.button_3.config(font='{Impact} 16 {}', justify='left', text='Image processing')
        self.button_3.pack(padx='5', pady='5', side='top')
        self.button_4 = tk.Button(self.frame_2)
        self.button_4.config(font='{Impact} 16 {}', justify='left', text='Go to grip')
        self.button_4.pack(padx='5', pady='5', side='top')
        self.button_5 = tk.Button(self.frame_2)
        self.button_5.config(font='{Impact} 16 {}', justify='left', text='HOME', command=demo1_home())
        self.button_5.pack(padx='5', pady='5', side='top')
        self.button_6 = tk.Button(self.frame_2)
        self.button_6.config(font='{Impact} 16 {}', justify='left', text='Grip')
        self.button_6.pack(padx='5', pady='5', side='top')
        self.button_7 = tk.Button(self.frame_2)
        self.button_7.config(font='{Impact} 16 {}', justify='left', text='Relerse')
        self.button_7.pack(padx='5', pady='5', side='top')
        self.button_8 = tk.Button(self.frame_2)
        self.button_8.config(font='{Impact} 16 {}', justify='left', text='Lnw Module')
        self.button_8.pack(padx='5', pady='5', side='top')
        self.button_9 = tk.Button(self.frame_2)
        self.button_9.config(activebackground='#0080ff', background='#ff0000', font='{Impact} 24 {}', foreground='#ffffff')
        self.button_9.config(justify='left', text='Do not click !!!!')
        self.button_9.pack(padx='5', pady='5', side='top')
        self.button_10 = tk.Button(self.frame_2)
        self.button_10.config(activebackground='#0080ff', background='#ff0000', bitmap='error', font='{Impact} 24 {}')
        self.button_10.config(foreground='#ffffff', justify='left')
        self.button_10.pack(padx='5', pady='5', side='top')
        self.frame_2.config(background='#ffff80', height='200', width='200')
        self.frame_2.pack(padx='10', pady='10', side='top')
        self.frame_1.config(background='#ffff00', height='200', width='200')
        self.frame_1.pack(side='top')

        # Main widget
        self.mainwindow = self.frame_1


    def run(self):
        self.mainwindow.mainloop()

if __name__ == '__main__':
    import tkinter as tk
    root = tk.Tk()
    app = LnwApp(root)
    app.run()
