import sqlite3
from tkinter import *
from tkinter.scrolledtext import *
from tkinter.font import Font
from tkinter import messagebox
import requests
from PIL import Image, ImageTk
from threading import Thread
import cv2
import base64
import numpy as np
import os
import subprocess


class Database:
    def __init__(self):
        self.create_db()

    def create_db(self):
        conn, cursor = self.connect_db()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
            )
        ''')
        self.commit_close(conn)

    def commit_close(self, conn):
        conn.commit()
        conn.close()

    def connect_db(self):
        conn = sqlite3.connect('userinfo.db')
        cursor = conn.cursor()
        return conn, cursor
    
    def insert_user(self, username, password):
        conn, cursor = self.connect_db()
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        self.commit_close(conn)

    def existing_user(self, username):
        conn, cursor = self.connect_db()
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        self.commit_close(conn)
        return result is not None  # false if user doesn't exist already, true if it does

    def get_password(self, username):
        conn, cursor = self.connect_db()
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))  # gets id corresponding to username
        id_number = cursor.fetchone()[0]
        cursor.execute('SELECT password FROM users WHERE id = ?', (id_number,))  # gets password corresponding to id
        password = cursor.fetchone()[0]
        self.commit_close(conn)
        return password


class GuiTkinter:
    def __init__(self, root):
        self.root = root
        self.root.title('user login')
        self.root.geometry('320x150')
        self.Database = Database()

        self.video_running = False
        self.video_paused = False
        self.cap = None
        self.login_page()

    def login_page(self):
        self.user_entry_text = StringVar()
        self.pw_entry_text = StringVar()

        entries_panel = Frame(self.root)
        entries_panel.grid(row=1, column=1, rowspan=3, padx=10, pady=5, sticky='NWSE')

        username_entry_label = Label(entries_panel, text='username: ')
        username_entry_label.grid(row=1, column=1, padx=5, pady=5)

        username_entry = Entry(entries_panel, textvariable=self.user_entry_text)
        username_entry.grid(row=1, column=2, padx=5, pady=5)

        password_entry_label = Label(entries_panel, text='password: ')
        password_entry_label.grid(row=2, column=1, padx=5, pady=5)

        password_entry = Entry(entries_panel, textvariable=self.pw_entry_text)
        password_entry.grid(row=2, column=2, padx=5, pady=5)

        buttons_panel = Frame(self.root)
        buttons_panel.grid(row=5, column=1, rowspan=1, padx=45, pady=5, sticky='NWSE')

        login_button = Button(buttons_panel, text='login', command=self.login)
        login_button.grid(row=1, column=1, ipadx=3, ipady=2, padx=5, pady=5)

        create_acc_button = Button(buttons_panel, text='create account', command=self.create_acc)
        create_acc_button.grid(row=1, column=2, ipadx=3, ipady=2, padx=5, pady=5)

    def login(self):
        username = self.user_entry_text.get()
        password = self.pw_entry_text.get()

        if not username or not password:  # makes sure entry fields are not blank
            messagebox.showinfo(message='one or more entries were left blank. please try again')
            return
        if not self.Database.existing_user(username):
            messagebox.showinfo(message='invalid username. please try again')
            return
        if password != password:
            messagebox.showinfo(message='invalid password. please try again')
            return
        
        messagebox.showinfo(message=f'login successful! welcome {username}')  # successful login!
        self.create_robot_gui()  # launches robot remote control window

    def create_acc(self):
        username = self.user_entry_text.get()
        password = self.pw_entry_text.get()

        if not username or not password:  # makes sure entries are not empty
            messagebox.showinfo(message='one or more entries were left blank. please try again')
            return
        if self.Database.existing_user(username):
            messagebox.showinfo(message='account with this username already exists.')
            return
        
        self.Database.insert_user(username, password)  # insert user and pw into database
        messagebox.showinfo(message=f'account creation successful!')

    def update_vid(self):
        def update_vid_stream(self):
            while self.video_running:
                if self.video_paused:
                    continue

                if not self.cap.isOpened():
                    print('uh oh')
                    break

                ret, frame = self.cap.read() 
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (480, 270))
                stream_img = Image.fromarray(frame_rgb)
                stream_imgtk = ImageTk.PhotoImage(image=stream_img)

                self.stream_elem.config(image=stream_imgtk)
                self.stream_elem.image = stream_imgtk

                self.overlay_elem.config(image=stream_imgtk)
                self.overlay_elem.image = stream_imgtk

                self.root.update_idletasks() 
            
            self.cap.release()
            self.cap = None
            self.video_running = False

    def play_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture('roadvid.mov')
            self.video_running = True
            self.video_paused = False

            self.video_thread = Thread(target=self.update_vid)
            self.video_thread.daemon = True
            self.video_thread.start() 
        else:
            self.video_paused = False

    def stop_video(self):
        self.video_paused = True

    def create_robot_gui(self):
        robot_gui = Toplevel()
        robot_gui.title('robot gui')
        robot_gui.geometry('1100x800')

        custom = Font(family='Poppins', size=20)

        vid_stream_panel = Frame(robot_gui)
        vid_stream_panel.grid(row=5, column=1, rowspan=1, padx=5, pady=5, sticky='NWSE')

        buttons_panel = Frame(robot_gui)
        buttons_panel.grid(row=5, column=2, rowspan=1, padx=20, pady=95, sticky='NWSE')

        vid_overlay_panel = Frame(robot_gui)
        vid_overlay_panel.grid(row=6, column=1, rowspan=1, padx=5, pady=5, sticky='NWSE')

        log_panel = Frame(robot_gui)
        log_panel.grid(row=6, column=2, rowspan=1, padx=10, pady=50, sticky='NS')

        text_area = ScrolledText(log_panel, width=55, height=5)
        text_area.grid(row=1, padx=5, pady=5, ipadx=20, ipady=20)
        text_area.config(state='disabled')

        open_file_button = Button(log_panel, text='open log file', command=self.open_log_file, font=custom, padx=5, pady=7)
        open_file_button.grid(row=2, padx=4, pady=5, ipadx=5, ipady=5,)

        self.stream_elem = Label(vid_stream_panel, text='vid stream')
        self.stream_elem.grid(padx=50, pady=40)

        self.overlay_elem = Label(vid_overlay_panel, text='vid overlay')
        self.overlay_elem.grid(padx=50, pady=10)


        forward = Button(buttons_panel, text='move forward', font=custom, padx=5, pady=7)
        forward.grid(row=1, column=2, padx=5, pady=5, ipadx=5, ipady=5, sticky='we', columnspan=2)
        
        left = Button(buttons_panel, text='move left', font=custom, padx=5, pady=7)
        left.grid(row=2, column=1, padx=2.5, pady=5, ipadx=5, ipady=5)
       
        play = Button(buttons_panel, text='play', font=custom, padx=5, pady=7, command = self.play_video)
        play.grid(row=2, column=2, padx=2.5, pady=5, ipadx=5, ipady=5, sticky='we')

        stop = Button(buttons_panel, text='stop', font=custom, padx=5, pady=7, command = self.stop_video) 
        stop.grid(row=2, column=3, padx=2.5, pady=5, ipadx=5, ipady=5, sticky='we')

        right = Button(buttons_panel, text='move right', font=custom, padx=5, pady=7)
        right.grid(row=2, column=4, padx=2.5, pady=5, ipadx=5, ipady=5, sticky='we')
        
        backward = Button(buttons_panel, text='move backward', font=custom, padx=5, pady=7)
        backward.grid(row=3, column=2, padx=2.5, pady=5, ipadx=5, ipady=5, sticky='we', columnspan=2)
        
    def open_log_file(self):
        file_path = 'system_log.txt'
        if not os.path.exists(file_path):
            open(file_path, 'w').close()
        subprocess.call(('open', file_path))

    def log_direction(self, direction, user):
        pass


if __name__ == '__main__':
    root = Tk()
    app = GuiTkinter(root)
    root.mainloop()
