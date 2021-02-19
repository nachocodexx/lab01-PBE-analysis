# Importar modulos


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)
plt.rcParams["figure.figsize"] = (20,10)
```

# Funciones de utileria


```python
def getTasa(time,corpus_size):
    corpus_size_bits = corpus_size*8
    corpus_size_mb   = corpus_size/125000
    tasa_bits_secs   = corpus_size_bits/time
    tasa_bytes_secs  = corpus_size/time
    tasa_mb_secs     = corpus_size_mb/time
    return {'BITS/SEC':(corpus_size_bits,tasa_bits_secs),'BYTES/SEC':(corpus_size,tasa_bytes_secs),'MB/SEC':(corpus_size_mb,tasa_mb_secs)}

def getInfo(**kwargs):
    dfa              = kwargs.get('df') 
    time             = dfa.TIME.sum()/1000
    time_min         = time/60
    corpus_size      = dfa.FILE_SIZE.sum()
    tasas            = getTasa(time,corpus_size)
    tasa_bits_secs   = tasas['BITS/SEC']
    tasa_bytes_secs  = tasas['BYTES/SEC']
    tasa_mb_secs     = tasas['MB/SEC']
    keyLen           = dfa.KEY_LENGTH.unique()[0]
    cipher           = dfa.CIPHER.unique()[0]+"[{}]".format(keyLen)
    
    return (cipher,time,time_min,*tasa_bits_secs,*tasa_bytes_secs,*tasa_mb_secs)
    
```

# Cipher(Encrypt mode)


```python
df      = pd.read_csv('data.csv')
des     = df.loc[df.CIPHER=='DES']
desede  = df.loc[df.CIPHER=='DESede']
aes_128 = df.loc[df.CIPHER=='AES'].loc[df.KEY_LENGTH==128]
aes_192 = df.loc[df.CIPHER=='AES'].loc[df.KEY_LENGTH==192]
aes_256 = df.loc[df.CIPHER=='AES'].loc[df.KEY_LENGTH==256].loc[df.KDF!='PBKDF2WithHmacSHA384'].loc[df.KDF!='PBKDF2WithHmacSHA512']
# 
aes_256_sha384 = df.loc[df.CIPHER=='AES'].loc[df.KEY_LENGTH==256].loc[df.KDF=='PBKDF2WithHmacSHA384']
df.loc[aes_256_sha384.index,'CIPHER']="AES_SHA384"
aes_256_sha384 = df.loc[df.CIPHER=='AES_SHA384'].loc[df.KEY_LENGTH==256].loc[df.KDF=='PBKDF2WithHmacSHA384']
# 
aes_256_sha512 = df.loc[df.CIPHER=='AES'].loc[df.KEY_LENGTH==256].loc[df.KDF=='PBKDF2WithHmacSHA512']
df.loc[aes_256_sha512.index,'CIPHER']="AES_SHA512"
aes_256_sha512 = df.loc[df.CIPHER=='AES_SHA512'].loc[df.KEY_LENGTH==256].loc[df.KDF=='PBKDF2WithHmacSHA512']
infos = list(map(lambda x:getInfo(df=x),(des,desede,aes_128,aes_192,aes_256,aes_256_sha384,aes_256_sha512)))
```

# Cipher(Decrypt mode)


```python
dff  = pd.read_csv('data_decrypted.csv')
dff.head(5)
_des     = dff.loc[dff.CIPHER=='DES']
_desede  = dff.loc[dff.CIPHER=='DESede']
_aes_128 = dff.loc[dff.CIPHER=='AES'].loc[dff.KEY_LENGTH==128]
_aes_192 = dff.loc[dff.CIPHER=='AES'].loc[dff.KEY_LENGTH==192]
_aes_256 = dff.loc[dff.CIPHER=='AES'].loc[dff.KEY_LENGTH==256].loc[dff.KDF!='PBKDF2WithHmacSHA384'].loc[df.KDF!='PBKDF2WithHmacSHA512']

# 
_aes_256_sha384 = dff.loc[df.CIPHER=='AES'].loc[dff.KEY_LENGTH==256].loc[dff.KDF=='PBKDF2WithHmacSHA384']
dff.loc[aes_256_sha384.index,'CIPHER']="AES_SHA384"
_aes_256_sha384 = dff.loc[dff.CIPHER=='AES_SHA384'].loc[dff.KEY_LENGTH==256].loc[dff.KDF=='PBKDF2WithHmacSHA384']
# 
_aes_256_sha512 = dff.loc[df.CIPHER=='AES'].loc[dff.KEY_LENGTH==256].loc[dff.KDF=='PBKDF2WithHmacSHA512']
dff.loc[aes_256_sha512.index,'CIPHER']="AES_SHA512"
_aes_256_sha512 = dff.loc[df.CIPHER=='AES_SHA512'].loc[dff.KEY_LENGTH==256].loc[dff.KDF=='PBKDF2WithHmacSHA512']
_infos = list(map(lambda x:getInfo(df=x),(_des,_desede,_aes_128,_aes_192,_aes_256,_aes_256_sha384,_aes_256_sha512)))
```

# RESULTADOS


```python
info_df = pd.DataFrame(infos,columns=["CIPHER","TIME(SEC)","TIME(MIN)","SIZE(BITS)","TASA(BITS/SEC)","SIZE(BYTES)","TASA(BYTES/SEC)","SIZE(MB)","TASA(MB/SEC)"])
info_df = info_df.sort_values("TIME(SEC)",ascending=True)
```

# Grafica 00


Muestra la cantidad de tiempo en segundos que tomo cada uno de los cifrados


```python
colors = ['#FF7C00','#1AC938','#E8000B','#8B2BE2','#F14CC1','#FFC400','#00D7FF']
sns.barplot(x="CIPHER",y="TIME(SEC)",data=info_df,palette=colors)
```




    <AxesSubplot:xlabel='CIPHER', ylabel='TIME(SEC)'>




    
![png](README_files/README_12_1.png)
    


# Grafica 01

Muestra la tasa en bits sobre segundo de cada uno de los cifrados en cifrar el corpus $\mathbb D$


```python
sns.barplot(x="CIPHER",y="TASA(BITS/SEC)",data=info_df,palette=colors)
```




    <AxesSubplot:xlabel='CIPHER', ylabel='TASA(BITS/SEC)'>




    
![png](README_files/README_15_1.png)
    


# Grafica 02

Muestra la tasa en Megabytes sobre segundo de cada uno de los cifrados


```python
sns.barplot(x="CIPHER",y="TASA(MB/SEC)",data=info_df,palette=colors)
```




    <AxesSubplot:xlabel='CIPHER', ylabel='TASA(MB/SEC)'>




    
![png](README_files/README_18_1.png)
    


# Decrypt Mode


```python
_info_df = pd.DataFrame(_infos,columns=["CIPHER","TIME(SEC)","TIME(MIN)","SIZE(BITS)","TASA(BITS/SEC)","SIZE(BYTES)","TASA(BYTES/SEC)","SIZE(MB)","TASA(MB/SEC)"])
_info_df = _info_df.sort_values("TIME(SEC)",ascending=True)
```

# Grafica 03

Muestra el tiempo en segundos que le toma decifrar el corpus $\mathbb D$


```python
colors2 =[colors[4],colors[3],colors[0],colors[2],colors[1],colors[6],colors[5]]
sns.barplot(x="CIPHER",y="TIME(SEC)",data=_info_df,palette=colors2)
```




    <AxesSubplot:xlabel='CIPHER', ylabel='TIME(SEC)'>




    
![png](README_files/README_23_1.png)
    


# Grafica 04

Muestra la tasa en megabytes por segundo que le toma a cada cifrador, decifrar el corpus $\mathbb D$


```python
sns.barplot(x="CIPHER",y="TASA(MB/SEC)",data=_info_df,palette=colors2)
```




    <AxesSubplot:xlabel='CIPHER', ylabel='TASA(MB/SEC)'>




    
![png](README_files/README_26_1.png)
    



```python
sns.color_palette("bright")
```




<svg  width="550" height="55"><rect x="0" y="0" width="55" height="55" style="fill:#023eff;stroke-width:2;stroke:rgb(255,255,255)"/><rect x="55" y="0" width="55" height="55" style="fill:#ff7c00;stroke-width:2;stroke:rgb(255,255,255)"/><rect x="110" y="0" width="55" height="55" style="fill:#1ac938;stroke-width:2;stroke:rgb(255,255,255)"/><rect x="165" y="0" width="55" height="55" style="fill:#e8000b;stroke-width:2;stroke:rgb(255,255,255)"/><rect x="220" y="0" width="55" height="55" style="fill:#8b2be2;stroke-width:2;stroke:rgb(255,255,255)"/><rect x="275" y="0" width="55" height="55" style="fill:#9f4800;stroke-width:2;stroke:rgb(255,255,255)"/><rect x="330" y="0" width="55" height="55" style="fill:#f14cc1;stroke-width:2;stroke:rgb(255,255,255)"/><rect x="385" y="0" width="55" height="55" style="fill:#a3a3a3;stroke-width:2;stroke:rgb(255,255,255)"/><rect x="440" y="0" width="55" height="55" style="fill:#ffc400;stroke-width:2;stroke:rgb(255,255,255)"/><rect x="495" y="0" width="55" height="55" style="fill:#00d7ff;stroke-width:2;stroke:rgb(255,255,255)"/></svg>


