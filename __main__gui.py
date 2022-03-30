import tkinter as tk
from tkinter import filedialog
import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def LINE__coord2direction(x,y,z):
    pl = np.rad2deg(-np.arcsin(z))
    tr = np.rad2deg(np.pi/2 - np.arctan2(y,x))
    return tr,pl
def LINE__direction2coord(tr,pl):
    tr,pl = np.deg2rad(tr),np.deg2rad(pl)
    x = np.cos(pl)*np.sin(tr)
    y = np.cos(pl)*np.cos(tr)
    z = np.sin(-pl)
    return x,y,z
def PSAxis_by_Vein(vein_strdip_data):
    # 各法線ベクトルデータについて、逆ベクトルも加えて分析
    # 目的は、データ平均を原点にすることで共分散行列の計算にnp.covを用いるため
    # この操作は主応力軸の方位及び主応力比に影響しない
    N = 2*vein_strdip_data.shape[0]
    # 岩脈の法線ベクトル格納
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    x[:N//2],y[:N//2],z[:N//2] = LINE__direction2coord(vein_strdip_data[:,0]+90, vein_strdip_data[:,1]-90)
    x[N//2:],y[N//2:],z[N//2:] = -x[:N//2],-y[:N//2],-z[:N//2]
    
    V = np.cov([x,y],z,bias=True)
    
    eig_val,eig_vec = LA.eig(V)
    eig_vec_sorted=eig_vec[:,np.argsort(eig_val)]
    sig1_vec = eig_vec_sorted[:,0]
    sig2_vec = eig_vec_sorted[:,1]
    sig3_vec = eig_vec_sorted[:,2]
    if sig1_vec[2]<0:
        sig1_vec = -sig1_vec
    if sig2_vec[2]<0:
        sig2_vec = -sig2_vec
    if sig3_vec[2]<0:
        sig3_vec = -sig3_vec
    
    return LINE__coord2direction(*sig1_vec),LINE__coord2direction(*sig2_vec),LINE__coord2direction(*sig3_vec)
def RST_calc(sig,Phi):
    tr = [sig[0][0],sig[1][0],sig[2][0]]
    pl = [sig[0][1],sig[1][1],sig[2][1]]
    P = np.array(LINE__direction2coord(tr, pl))
    tensor_SIGSystem = np.zeros((3,3))
    tensor_SIGSystem[0,0]=1
    tensor_SIGSystem[1,1]=Phi
    tensor = P@tensor_SIGSystem@P.T
    return tensor
def Sn_Ss_calc(vein_strdip_data,Phi):
    tensor = RST_calc(PSAxis_by_Vein(vein_strdip_data),Phi)
    # 任意の面にはたらく法線応力、剪断応力の大きさは、単位法線ベクトルのとり方に依存しない。
    # データを複製する必要はない。
    N = vein_strdip_data.shape[0]
    # 岩脈の法線ベクトル格納
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    t = np.zeros((N,3))
    Sn = np.zeros(N)
    Ss = np.zeros(N)
    x,y,z = LINE__direction2coord(vein_strdip_data[:,0]+90, vein_strdip_data[:,1]-90)
    for i in range(N):
        t[i,:] = np.dot(tensor,[x[i],y[i],z[i]])
        Sn[i] = np.dot(t[i,:],[x[i],y[i],z[i]])
        t2 = np.dot(t[i,:],t[i,:])
        Ss[i] = np.sqrt(t2-Sn[i]**2)
    return Sn,Ss
def Mohr(vein_strdip_data,Phi):
    Sn,Ss = Sn_Ss_calc(vein_strdip_data, Phi)
    Sn_max = np.max(Sn)
    # fig,ax = plt.subplots()
    fig = plt.Figure()
    ax = fig.add_subplot()
    ax.scatter(Sn,Ss)
    ax.add_patch(patches.Circle(xy=(0.5,0),radius=0.5,fill=False))
    ax.add_patch(patches.Circle(xy=(Phi/2,0),radius=Phi/2,fill=False))
    ax.add_patch(patches.Circle(xy=((Phi+1)/2,0),radius=(1-Phi)/2,fill=False))
    ax.plot([Sn_max,Sn_max],[0,0.6])
    
    ax.text(Sn_max,0.63,r"$\lambda'=\frac{P_f-\sigma_3}{\sigma_1-\sigma_3}=$%.3f"%Sn_max,ha='center')
    if Phi>0.8:
        ax.text(0.5,-0.25,r'$\Phi=$%.3f'%Phi + 'であり、軸性引張に近い。',family='meiryo',transform=ax.transAxes,ha='center',va='center')
    elif Phi<0.2:
        ax.text(0.5,-0.25,r'$\Phi=$%.3f'%Phi + 'であり、軸性圧縮に近い。',family='meiryo',transform=ax.transAxes,ha='center',va='center')
    else:
        ax.text(0.5,-0.25,r'$\Phi=$%.3f'%Phi + 'であり、三軸応力と言える。',family='meiryo',transform=ax.transAxes,ha='center',va='center')
        
    ax.set_ylim(0,0.6)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\sigma_n$')
    ax.set_ylabel(r'$\sigma_s$')
    fig.subplots_adjust(top=0.9,bottom=0.3,left=0.1,right=0.9)
    return fig
def PSratio_auto_Spearman(vein_strdip_data,N):
    unifP_data = np.loadtxt('unifP1000.csv',delimiter=',')
    for i in range(unifP_data.shape[0]):
        unifP_data[i] = unifP_data[i][0]-90, unifP_data[i][1]+90
    # 応力比の刻み幅
    rho = np.zeros(N+1)
    # binの数
    NG = 10
    for i in range(N+1):
        Phi = i/N
        freq = np.histogram(Sn_Ss_calc(vein_strdip_data, Phi)[0],bins=np.arange(NG+1)/NG)[0]
        freq_unif = np.histogram(Sn_Ss_calc(unifP_data, Phi)[0],bins=np.arange(NG+1)/NG)[0]
        freq_STD = freq/freq_unif
        rho[i]=np.sum((NG-np.argsort(np.argsort(freq_STD))-np.arange(NG))**2)
        rho[i]=1-6*rho[i]/(NG*(NG**2-1))
    return np.argmax(rho)/N,rho
def Scatter_rho(rho):
    N = len(rho)
    x = np.arange(N)/(N-1)
    Phi = x[np.argmax(rho)]
    # fig,ax = plt.subplots()
    fig = plt.Figure()
    ax = fig.add_subplot()
    ax.scatter(x,rho)
    ax.plot([Phi,Phi],[-0.5,1.5])
    ax.text(Phi,1.05,r'$\Phi=%.2f$'%Phi,ha='center')
    ax.set_ylim(0,1)
    ax.set_xlabel(r'$\Phi$')
    ax.set_ylabel(r'$\rho$')
    return fig


app = tk.Tk()
app.geometry('1000x1000')
app.title('StressAnalyze_by_Dyke')
app.resizable(width=False,height=False)

def open_func():
    filetype = [('csvファイル','*.csv')]
    filepath = tk.filedialog.askopenfilename(filetypes = filetype)
    if filepath=='':
        pass
    else:
        label_filepass['text']=filepath

def execute_func():
    if label_filepass['text']=='no file':
        label_execute['text']='no file'
    else:
        try:
            df = pd.read_csv(label_filepass['text'],header=None)
            data_pre = df.to_numpy()
            data=np.zeros((data_pre.shape[0],2))
            for i in range(data_pre.shape[0]):
                data[i,:]=np.array(data_pre[i,0].split(','),dtype=np.float64)
        except:
            data = np.loadtxt(label_filepass['text'],delimiter=',')
        n = data.shape[0]
        v1,v2,v3=PSAxis_by_Vein(data)
        Phi,rho = PSratio_auto_Spearman(data, 100)
        label_execute['text']='n=%d\n'%n
        label_execute['text']+='\n'
        label_execute['text']+='principal stress axes:\n'
        label_execute['text']+='az=%.1f %.1f %.1f\n'%(v1[0],v2[0],v3[0])
        label_execute['text']+='pl=%.1f %.1f %.1f\n'%(v1[1],v2[1],v3[1])
        label_execute['text']+='\n'
        label_execute['text']+='optimal Phi=%.3f\n'%Phi
        
        fig_Mohr = Mohr(data, Phi)
        fig_rho = Scatter_rho(rho)
        canvas1 = FigureCanvasTkAgg(fig_Mohr,app)
        canvas1.draw()
        canvas1.get_tk_widget().place(relx=0.32,rely=0.2,relwidth=0.66,relheight=0.38)
        canvas2 = FigureCanvasTkAgg(fig_rho,app)
        canvas2.draw()
        canvas2.get_tk_widget().place(relx=0.32,rely=0.6,relwidth=0.66,relheight=0.38)
        

field_file_pass = tk.Canvas(app,bg='#98FB98')
label_filepass = tk.Label(app,bg='#98FB98',font=('meiryo',16),text='no file',wraplength=550)

btn_open = tk.Button(app,bg='#aaa',text='open',font=('meiryo',20),command=open_func)

btn_execute = tk.Button(app,bg='#aaa',text='execute',font=('meiryo',20),command=execute_func)

field_execute = tk.Canvas(app)
label_execute = tk.Label(app,font=('meiryo',12),text='no file',wraplength=240,)

# fig,ax = plt.subplots()
fig = plt.Figure()
ax = fig.add_subplot()

field_Mohr = FigureCanvasTkAgg(fig,app)
field_rho = FigureCanvasTkAgg(fig,app)


# 配置

field_file_pass.place(relx=0.2,rely=0.02,relwidth=0.6,relheight=0.16)
label_filepass.place(relx=0.5,rely=0.1,anchor='center')

btn_open.place(relx=0.82,rely=0.04,relwidth=0.13,relheight=0.12)

btn_execute.place(relx=0.02,rely=0.2,relwidth=0.28,relheight=0.18)
label_execute.place(relx=0.16,rely=0.42,anchor='n')

field_execute.place(relx=0.02,rely=0.4,relwidth=0.28,relheight=0.58)

field_Mohr.get_tk_widget().place(relx=0.32,rely=0.2,relwidth=0.66,relheight=0.38)
field_rho.get_tk_widget().place(relx=0.32,rely=0.6,relwidth=0.66,relheight=0.38)

app.mainloop()

# テストを書いてみてます。