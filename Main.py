import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd
from tkinter import *
import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm
import clusterformation as cfor
import prediction as kmeans

bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"


def Home():
        global window
        def clear():
            print("Clear1")
            txt.delete(0, 'end')  
            txt1.delete(0, 'end')    
            txt3.delete(0, 'end')      
            
  



        window = tk.Tk()
        window.title("NETWORK ANOMALY DETECTION USING K-MEANS CLUSTERING")
        
 
        window.geometry('1580x960')
        window.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        

        message1 = tk.Label(window, text="NETWORK ANOMALY DETECTION USING K-MEANS CLUSTERING" ,bg=bgcolor  ,fg=fgcolor  ,width=80  ,height=2,font=('times', 20, 'italic bold underline')) 
        message1.place(x=100, y=1)

        lbl = tk.Label(window, text="Train Dataset",width=20  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl.place(x=100, y=100)
        
        txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt.place(x=400, y=110)
        lbl1 = tk.Label(window, text="Test Dataset",width=20  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1.place(x=100, y=200)
        
        txt1 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1.place(x=400, y=210)
        lbl3 = tk.Label(window, text="Enter No of Cluster",width=20  ,height=1  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl3.place(x=100, y=300)
        
        txt3 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt3.place(x=400, y=310)
        

        


        def browse():
                path=filedialog.askopenfilename()
                print(path)
                txt.delete(0, 'end')
                txt.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Training Datset") 
        def browse1():
                path=filedialog.askopenfilename()
                print(path)
                txt1.delete(0, 'end')
                txt1.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Test Datset")     

        
        def preproc():
                tr_path=txt.get()
                te_path=txt1.get()
                nofc=int(txt3.get())

                cfor.process(tr_path,te_path,nofc)
                print("Cluster Formation")
                tm.showinfo("Input", "Cluster Formation Successfully Finished")
                
        def LRprocess():
                tr_path=txt.get()
                te_path=txt1.get()
                nofc=int(txt3.get())
                if tr_path != "" and te_path !="":
                        kmeans.process(tr_path,te_path,nofc)
                        tm.showinfo("Input", "K-means Prediction Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset File")

        browsebtn = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browsebtn.place(x=600, y=110)
        browsebtn1 = tk.Button(window, text="Browse", command=browse1  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browsebtn1.place(x=600, y=210)

        clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=14  ,height=1 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        clearButton.place(x=500, y=500)
         
        proc = tk.Button(window, text="Cluster Formation", command=preproc  ,fg=fgcolor   ,bg=bgcolor1   ,width=14  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        proc.place(x=100, y=500)
        

        LRbutton = tk.Button(window, text="Prediction", command=LRprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=14  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        LRbutton.place(x=300, y=500)


        
        

        quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=700, y=500)

        window.mainloop()
Home()

