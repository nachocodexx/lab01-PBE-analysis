{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 467,
   "id": "gross-default",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "sns.set_theme(style=\"ticks\", color_codes=True)\n",
    "plt.rcParams[\"figure.figsize\"] = (20,10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 466,
   "id": "located-summit",
   "metadata": {},
   "outputs": [],
   "source": [
    "def getTasa(time,corpus_size):\n",
    "    corpus_size_bits = corpus_size*8\n",
    "    corpus_size_mb   = corpus_size/125000\n",
    "    tasa_bits_secs   = corpus_size_bits/time\n",
    "    tasa_bytes_secs  = corpus_size/time\n",
    "    tasa_mb_secs     = corpus_size_mb/time\n",
    "    return {'BITS/SEC':(corpus_size_bits,tasa_bits_secs),'BYTES/SEC':(corpus_size,tasa_bytes_secs),'MB/SEC':(corpus_size_mb,tasa_mb_secs)}\n",
    "def getInfo(**kwargs):\n",
    "    dfa              = kwargs.get('df') \n",
    "    time             = dfa.TIME.sum()/1000\n",
    "    time_min         = time/60\n",
    "    corpus_size      = dfa.FILE_SIZE.sum()\n",
    "    tasas            = getTasa(time,corpus_size)\n",
    "    tasa_bits_secs   = tasas['BITS/SEC']\n",
    "    tasa_bytes_secs  = tasas['BYTES/SEC']\n",
    "    tasa_mb_secs     = tasas['MB/SEC']\n",
    "    keyLen           = dfa.KEY_LENGTH.unique()[0]\n",
    "    cipher           = dfa.CIPHER.unique()[0]+\"[{}]\".format(keyLen)\n",
    "    \n",
    "    return (cipher,time,time_min,*tasa_bits_secs,*tasa_bytes_secs,*tasa_mb_secs)\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "sealed-terror",
   "metadata": {},
   "source": [
    "# Cipher(Encrypt mode)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 474,
   "id": "peripheral-concept",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>FILE_NAME</th>\n",
       "      <th>FILE_SIZE</th>\n",
       "      <th>TIME</th>\n",
       "      <th>KDF</th>\n",
       "      <th>CIPHER</th>\n",
       "      <th>TRANSFORMATION</th>\n",
       "      <th>KEY_LENGTH</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>4704</th>\n",
       "      <td>650.txt</td>\n",
       "      <td>1048575</td>\n",
       "      <td>375</td>\n",
       "      <td>PBKDF2WithHmacSHA256</td>\n",
       "      <td>AES</td>\n",
       "      <td>AES/CBC/PKCS5Padding</td>\n",
       "      <td>256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4705</th>\n",
       "      <td>916.txt</td>\n",
       "      <td>1048575</td>\n",
       "      <td>113</td>\n",
       "      <td>PBKDF2WithHmacSHA256</td>\n",
       "      <td>AES</td>\n",
       "      <td>AES/CBC/PKCS5Padding</td>\n",
       "      <td>256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4706</th>\n",
       "      <td>276.txt</td>\n",
       "      <td>1048575</td>\n",
       "      <td>90</td>\n",
       "      <td>PBKDF2WithHmacSHA256</td>\n",
       "      <td>AES</td>\n",
       "      <td>AES/CBC/PKCS5Padding</td>\n",
       "      <td>256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4707</th>\n",
       "      <td>1483.txt</td>\n",
       "      <td>1048575</td>\n",
       "      <td>69</td>\n",
       "      <td>PBKDF2WithHmacSHA256</td>\n",
       "      <td>AES</td>\n",
       "      <td>AES/CBC/PKCS5Padding</td>\n",
       "      <td>256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4708</th>\n",
       "      <td>1122.txt</td>\n",
       "      <td>1048575</td>\n",
       "      <td>64</td>\n",
       "      <td>PBKDF2WithHmacSHA256</td>\n",
       "      <td>AES</td>\n",
       "      <td>AES/CBC/PKCS5Padding</td>\n",
       "      <td>256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5875</th>\n",
       "      <td>518.txt</td>\n",
       "      <td>1048575</td>\n",
       "      <td>44</td>\n",
       "      <td>PBKDF2WithHmacSHA256</td>\n",
       "      <td>AES</td>\n",
       "      <td>AES/CBC/PKCS5Padding</td>\n",
       "      <td>256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5876</th>\n",
       "      <td>115.txt</td>\n",
       "      <td>1048575</td>\n",
       "      <td>51</td>\n",
       "      <td>PBKDF2WithHmacSHA256</td>\n",
       "      <td>AES</td>\n",
       "      <td>AES/CBC/PKCS5Padding</td>\n",
       "      <td>256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5877</th>\n",
       "      <td>512.txt</td>\n",
       "      <td>1048575</td>\n",
       "      <td>48</td>\n",
       "      <td>PBKDF2WithHmacSHA256</td>\n",
       "      <td>AES</td>\n",
       "      <td>AES/CBC/PKCS5Padding</td>\n",
       "      <td>256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5878</th>\n",
       "      <td>643.txt</td>\n",
       "      <td>1048575</td>\n",
       "      <td>51</td>\n",
       "      <td>PBKDF2WithHmacSHA256</td>\n",
       "      <td>AES</td>\n",
       "      <td>AES/CBC/PKCS5Padding</td>\n",
       "      <td>256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5879</th>\n",
       "      <td>699.txt</td>\n",
       "      <td>1048575</td>\n",
       "      <td>47</td>\n",
       "      <td>PBKDF2WithHmacSHA256</td>\n",
       "      <td>AES</td>\n",
       "      <td>AES/CBC/PKCS5Padding</td>\n",
       "      <td>256</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1176 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     FILE_NAME  FILE_SIZE  TIME                   KDF CIPHER  \\\n",
       "4704   650.txt    1048575   375  PBKDF2WithHmacSHA256    AES   \n",
       "4705   916.txt    1048575   113  PBKDF2WithHmacSHA256    AES   \n",
       "4706   276.txt    1048575    90  PBKDF2WithHmacSHA256    AES   \n",
       "4707  1483.txt    1048575    69  PBKDF2WithHmacSHA256    AES   \n",
       "4708  1122.txt    1048575    64  PBKDF2WithHmacSHA256    AES   \n",
       "...        ...        ...   ...                   ...    ...   \n",
       "5875   518.txt    1048575    44  PBKDF2WithHmacSHA256    AES   \n",
       "5876   115.txt    1048575    51  PBKDF2WithHmacSHA256    AES   \n",
       "5877   512.txt    1048575    48  PBKDF2WithHmacSHA256    AES   \n",
       "5878   643.txt    1048575    51  PBKDF2WithHmacSHA256    AES   \n",
       "5879   699.txt    1048575    47  PBKDF2WithHmacSHA256    AES   \n",
       "\n",
       "            TRANSFORMATION  KEY_LENGTH  \n",
       "4704  AES/CBC/PKCS5Padding         256  \n",
       "4705  AES/CBC/PKCS5Padding         256  \n",
       "4706  AES/CBC/PKCS5Padding         256  \n",
       "4707  AES/CBC/PKCS5Padding         256  \n",
       "4708  AES/CBC/PKCS5Padding         256  \n",
       "...                    ...         ...  \n",
       "5875  AES/CBC/PKCS5Padding         256  \n",
       "5876  AES/CBC/PKCS5Padding         256  \n",
       "5877  AES/CBC/PKCS5Padding         256  \n",
       "5878  AES/CBC/PKCS5Padding         256  \n",
       "5879  AES/CBC/PKCS5Padding         256  \n",
       "\n",
       "[1176 rows x 7 columns]"
      ]
     },
     "execution_count": 474,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df      = pd.read_csv('data.csv')\n",
    "des     = df.loc[df.CIPHER=='DES']\n",
    "desede  = df.loc[df.CIPHER=='DESede']\n",
    "aes_128 = df.loc[df.CIPHER=='AES'].loc[df.KEY_LENGTH==128]\n",
    "aes_192 = df.loc[df.CIPHER=='AES'].loc[df.KEY_LENGTH==192]\n",
    "aes_256 = df.loc[df.CIPHER=='AES'].loc[df.KEY_LENGTH==256].loc[df.KDF!='PBKDF2WithHmacSHA384'].loc[df.KDF!='PBKDF2WithHmacSHA512']\n",
    "# \n",
    "aes_256_sha384 = df.loc[df.CIPHER=='AES'].loc[df.KEY_LENGTH==256].loc[df.KDF=='PBKDF2WithHmacSHA384']\n",
    "df.loc[aes_256_sha384.index,'CIPHER']=\"AES_SHA384\"\n",
    "aes_256_sha384 = df.loc[df.CIPHER=='AES_SHA384'].loc[df.KEY_LENGTH==256].loc[df.KDF=='PBKDF2WithHmacSHA384']\n",
    "# \n",
    "aes_256_sha512 = df.loc[df.CIPHER=='AES'].loc[df.KEY_LENGTH==256].loc[df.KDF=='PBKDF2WithHmacSHA512']\n",
    "df.loc[aes_256_sha512.index,'CIPHER']=\"AES_SHA512\"\n",
    "aes_256_sha512 = df.loc[df.CIPHER=='AES_SHA512'].loc[df.KEY_LENGTH==256].loc[df.KDF=='PBKDF2WithHmacSHA512']\n",
    "infos = list(map(lambda x:getInfo(df=x),(des,desede,aes_128,aes_192,aes_256,aes_256_sha384,aes_256_sha512)))\n",
    "aes_256"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "manufactured-bibliography",
   "metadata": {},
   "source": [
    "# Cipher(Decrypt mode)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 482,
   "id": "integrated-thread",
   "metadata": {},
   "outputs": [],
   "source": [
    "dff  = pd.read_csv('data_decrypted.csv')\n",
    "dff.head(5)\n",
    "_des     = dff.loc[dff.CIPHER=='DES']\n",
    "_desede  = dff.loc[dff.CIPHER=='DESede']\n",
    "_aes_128 = dff.loc[dff.CIPHER=='AES'].loc[dff.KEY_LENGTH==128]\n",
    "_aes_192 = dff.loc[dff.CIPHER=='AES'].loc[dff.KEY_LENGTH==192]\n",
    "_aes_256 = dff.loc[dff.CIPHER=='AES'].loc[dff.KEY_LENGTH==256].loc[dff.KDF!='PBKDF2WithHmacSHA384'].loc[df.KDF!='PBKDF2WithHmacSHA512']\n",
    "\n",
    "\n",
    "\n",
    "# \n",
    "_aes_256_sha384 = dff.loc[df.CIPHER=='AES'].loc[dff.KEY_LENGTH==256].loc[dff.KDF=='PBKDF2WithHmacSHA384']\n",
    "dff.loc[aes_256_sha384.index,'CIPHER']=\"AES_SHA384\"\n",
    "_aes_256_sha384 = dff.loc[dff.CIPHER=='AES_SHA384'].loc[dff.KEY_LENGTH==256].loc[dff.KDF=='PBKDF2WithHmacSHA384']\n",
    "# \n",
    "_aes_256_sha512 = dff.loc[df.CIPHER=='AES'].loc[dff.KEY_LENGTH==256].loc[dff.KDF=='PBKDF2WithHmacSHA512']\n",
    "dff.loc[aes_256_sha512.index,'CIPHER']=\"AES_SHA512\"\n",
    "_aes_256_sha512 = dff.loc[df.CIPHER=='AES_SHA512'].loc[dff.KEY_LENGTH==256].loc[dff.KDF=='PBKDF2WithHmacSHA512']\n",
    "_infos = list(map(lambda x:getInfo(df=x),(_des,_desede,_aes_128,_aes_192,_aes_256,_aes_256_sha384,_aes_256_sha512)))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aboriginal-tomorrow",
   "metadata": {},
   "source": [
    "# RESULTADOS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 483,
   "id": "decreased-runner",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CIPHER</th>\n",
       "      <th>TIME(SEC)</th>\n",
       "      <th>TIME(MIN)</th>\n",
       "      <th>SIZE(BITS)</th>\n",
       "      <th>TASA(BITS/SEC)</th>\n",
       "      <th>SIZE(BYTES)</th>\n",
       "      <th>TASA(BYTES/SEC)</th>\n",
       "      <th>SIZE(MB)</th>\n",
       "      <th>TASA(MB/SEC)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>AES[128]</td>\n",
       "      <td>122.400</td>\n",
       "      <td>2.040000</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>1.688335e+08</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>2.110419e+07</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>168.833502</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>AES_SHA512[256]</td>\n",
       "      <td>127.299</td>\n",
       "      <td>2.121650</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>1.623361e+08</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>2.029201e+07</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>162.336080</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>AES[192]</td>\n",
       "      <td>133.092</td>\n",
       "      <td>2.218200</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>1.552702e+08</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>1.940877e+07</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>155.270194</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>AES[256]</td>\n",
       "      <td>137.553</td>\n",
       "      <td>2.292550</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>1.502346e+08</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>1.877933e+07</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>150.234605</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>AES_SHA384[256]</td>\n",
       "      <td>140.400</td>\n",
       "      <td>2.340000</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>1.471882e+08</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>1.839852e+07</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>147.188181</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>DES[64]</td>\n",
       "      <td>202.504</td>\n",
       "      <td>3.375067</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>1.020485e+08</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>1.275606e+07</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>102.048457</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>DESede[192]</td>\n",
       "      <td>396.793</td>\n",
       "      <td>6.613217</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>5.208061e+07</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>6.510076e+06</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>52.080608</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            CIPHER  TIME(SEC)  TIME(MIN)   SIZE(BITS)  TASA(BITS/SEC)  \\\n",
       "2         AES[128]    122.400   2.040000  20665220656    1.688335e+08   \n",
       "6  AES_SHA512[256]    127.299   2.121650  20665220656    1.623361e+08   \n",
       "3         AES[192]    133.092   2.218200  20665220656    1.552702e+08   \n",
       "4         AES[256]    137.553   2.292550  20665220656    1.502346e+08   \n",
       "5  AES_SHA384[256]    140.400   2.340000  20665220656    1.471882e+08   \n",
       "0          DES[64]    202.504   3.375067  20665220656    1.020485e+08   \n",
       "1      DESede[192]    396.793   6.613217  20665220656    5.208061e+07   \n",
       "\n",
       "   SIZE(BYTES)  TASA(BYTES/SEC)      SIZE(MB)  TASA(MB/SEC)  \n",
       "2   2583152582     2.110419e+07  20665.220656    168.833502  \n",
       "6   2583152582     2.029201e+07  20665.220656    162.336080  \n",
       "3   2583152582     1.940877e+07  20665.220656    155.270194  \n",
       "4   2583152582     1.877933e+07  20665.220656    150.234605  \n",
       "5   2583152582     1.839852e+07  20665.220656    147.188181  \n",
       "0   2583152582     1.275606e+07  20665.220656    102.048457  \n",
       "1   2583152582     6.510076e+06  20665.220656     52.080608  "
      ]
     },
     "execution_count": 483,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "info_df = pd.DataFrame(infos,columns=[\"CIPHER\",\"TIME(SEC)\",\"TIME(MIN)\",\"SIZE(BITS)\",\"TASA(BITS/SEC)\",\"SIZE(BYTES)\",\"TASA(BYTES/SEC)\",\"SIZE(MB)\",\"TASA(MB/SEC)\"])\n",
    "info_df = info_df.sort_values(\"TIME(SEC)\",ascending=True)\n",
    "info_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 491,
   "id": "sealed-clothing",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='CIPHER', ylabel='TIME(SEC)'>"
      ]
     },
     "execution_count": 491,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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lh/96Eeb3bahoSGLFy9u9qe+vn67jR8AAACAlrXJ5XL/qKmpKbfddlsGDBhQXnb66aensbExH/rQh3LBBRekqqoqS5cuzd57713eZu+9987SpUs3+5rTp0/P1KlTW33sAAAAAGxem0emr33ta9l5551z2mmnJUl+9rOfpba2NqtXr86ll16a66+/PhdffPE2vebIkSMzdOjQZsvq6+szYsSI7TZuAAAAALasTSPThAkT8te//jU33HBD+UbftbW1SZJdd901p5xySm655Zby8gcffLD83CVLlpS3fa3q6upUV1e38ugBAAAA2JI2uSdTknz961/PwoULc/3116eqqipJ8ve//z1r165NkmzcuDHz589P3759kyRHH310fv/73+eZZ55Jsunm4B//+MfbargAAAAAbIM2OZPpT3/6U2688cbst99+GTZsWJKkZ8+eOeecczJ69OhUVFRk48aN6d+/fy666KIkm85sGjt2bD73uc+lqakpffv2zVVXXdUWwwUAAABgG7VJZHr/+9+fJ554YrPr5s6du8XnffSjH81HP/rR1hoWAAAAANtJm10uBwAAAMBbl8gEAAAAQGEiEwAAAACFiUwAAAAAFCYyAQAAAFCYyAQAAABAYSITAAAAAIWJTAAAAAAUJjIBAAAAUJjIBAAAAEBhIhMAAAAAhYlMAAAAABQmMgEAAABQmMgEAAAAQGEiEwAAAACFiUwAAAAAFCYyAQAAAFCYyAQAAABAYSITAAAAAIWJTAAAAAAU1nlrNtqwYUOefvrpNDQ0pLq6Ou9973vTpUuX1h4bAAAAAB1Ei5HpZz/7WWbOnJn//u//TufOnbPLLrtkzZo12bhxY4444ogMGzYsH/nIR9pqrAAAAADsoLYYmYYNG5Zu3bpl0KBB+epXv5o999yzvG7ZsmV56KGHMnPmzNx4442ZOXNmmwwWAAAAgB3TFiPTV7/61fTp02ez6/bcc88MGjQogwYNyhNPPNFqgwMAAACgY9jijb+3FJje7HYAAAAAvHW1+NvlZs+enYsvvniz6y655JLcddddrTIoAAAAADqWFiPTzJkzc+6552523Wc/+9l8//vfb5VBAQAAANCxtBiZ/vrXv+aAAw7Y7Lr9998/zzzzTGuMCQAAAIAOpsXI1NTUlFWrVm123apVq9LU1NQaYwIAAACgg2kxMvXv3z+zZs3a7LrZs2fnkEMOaY0xAQAAANDBdG5p5fnnn5+RI0dm6dKlOe644/Kud70rL7zwQn784x9n9uzZmT59eluNEwAAAIAdWIuR6aCDDsp3vvOdTJo0Kd///vfT1NSUysrKHHLIIbn55pvzT//0T201TgAAAAB2YC1GpmTTJXPf//73s3bt2vz9739Pt27d8o53vKMtxgYAAABAB9HiPZl+9KMflf/7He94R9asWdMsMH33u99ttYEBAAAA0HG0GJmuuuqqZo+HDRvW7PGUKVO2/4gAAAAA6HBajEylUmmbHgMAAADw9tRiZKqoqNimxwAAAAC8Pb3hjb9LpVL5z+YeAwAAAECLkenll1/OAQccUH5cKpXKj0ulkjOZAAAAAEjyBpHpvvvua6txAAAAANCBtRiZevTosdnlf//739OtW7dWGRAAAAAAHU+LN/6eM2dOfvGLX5Qf//73v8+HP/zhHHHEETn++OPzl7/8pdUHCAAAAMCOr8XIdPPNN+dd73pX+fHo0aNz1FFH5Yc//GGOOuqoTJw4sdUHCAAAAMCOr8XL5err69O7d+8kydKlS/Pkk0/mlltuSU1NTb70pS/luOOOa5NBAgAAALBja/FMpk6dOmXDhg1Jksceeyzve9/7UlNTkyTZaaedsnbt2lYfIAAAAAA7vhYj0z//8z/n//yf/5M//vGPufXWW/ORj3ykvO4vf/lLs0vpAAAAAHj7ajEyXXXVVVm0aFE+85nPZKeddsq5555bXnfXXXfl6KOPbvUBAgAAALDja/GeTHvuuWf+4z/+Y7PrRo0a1SoDAgAAAKDj2eKZTC+++OJWvcDWbgcAAADAW9cWz2QaOXJkPvCBD2TIkCE5+OCDU1n5/3pUU1NTfve732XOnDl5+OGHM2/evDYZLAAAAAA7pi1GpjvvvDO33357vvKVr2Tx4sXZZ599sssuu2TNmjVZvHhx3vOe9+TUU0/NlVde2ZbjBQAAAGAHtMXIVFVVldNOOy2nnXZali5dmieffDINDQ2prq7O/vvvnz333LMtxwkAAADADqzFG3+/qra2NrW1ta09FgAAAAA6qC3e+DtJzjvvvGaPp0yZ0uzxpz71qe0/IgAAAAA6nBYj04MPPtjs8fe+971mj//yl79s/xEBAAAA0OG0GJleq1QqNXtcUVGxXQcDAAAAQMe0TZHpzUallStX5txzz83xxx+fE088Meeff35WrFiRJHn88cczePDgHH/88TnrrLOyfPny8vNaWgcAANDRbdywob2HQDvxveetqMUbf2/cuDGzZs0qn8G0fv363HHHHeX1jY2NW7WTioqKnHPOOTn88MOTJBMmTMjkyZNz7bXX5tJLL8348eNTV1eXadOmZfLkyRk/fnyampq2uA4AAOCtoHOXLvn6FZ9r72HQDi4Zf2N7DwG2uxYj08EHH5w5c+aUH//TP/1T7rrrrvLjgw46aKt2UlNTUw5MSXLIIYfktttuy8KFC9O1a9fU1dUlSYYNG5Zjjz0248ePb3HdazU0NKShoaHZsvr6+q0aGwAAAADFtRiZbr311u2+w6amptx2220ZMGBAli5dmr333ru8rnv37mlqasqqVataXFdTU9PsNadPn56pU6du97ECAAAAsHVajEyt4Wtf+1p23nnnnHbaafnJT36yXV5z5MiRGTp0aLNl9fX1GTFixHZ5fQAAAABa1mJk6tu37xbXlUqlVFRU5A9/+MNW72zChAn561//mhtuuCGVlZWpra3NkiVLyutXrFiRysrK1NTUtLjutaqrq1NdXb3V4wAAAABg+2oxMtXU1KRbt24ZOnRojj322FRVVb3pHX3961/PwoULc9NNN5Vfp1+/flm7dm0efvjh1NXVZebMmTnhhBPecB0AAAAAO5YWI9MvfvGL3H///ZkzZ05uvfXWDBgwIEOGDMlhhx22TTv505/+lBtvvDH77bdfhg0bliTp2bNnrr/++kycODFjxozJunXr0qNHj0yaNClJUllZucV1AAAAAOxYWoxMnTt3zoABAzJgwIA0NDTk7rvvzuTJk7NixYpMmzYtvXr12qqdvP/9788TTzyx2XWHHnpo5s6du83rAAAAANhxVG71hpWVqaioSJI0Nja22oAAAAAA6HhaPJOpqakp999/f+688848/PDDGTBgQL70pS+lrq6urcYHAAAAQAfQYmQ6+uijU11dnSFDhuSCCy5I165dkyTPPvtseZt99tmndUcIAAAAwA6vxci0fPnyLF++PP/3//7ffOMb30iSlEql8vqKior84Q9/aN0RAgAAALDDazEy/fGPf2yrcQAAAADQgW31jb8BAAAAYEtaPJPp0ksvLf9GuS2ZOHHidh0QAAAAAB1Pi5Fp3333batxAAAAANCBtRiZ9ttvvwwaNKit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G5dBgwbl+OOPb/ZzyL333puTTjopgwYNyic+8Yl85zvfKT/vnnvuycCBA1/3c02HV6LD+t73vlc67bTTSkcccURp3bp1pVKpVJo1a1bpggsu2Oz211xzTem6664rNTU1lUqlUmnFihWl5557rlQqlUpTpkwpXXfddc22//Wvf1168cUXS7179y6tXr26vPzZZ58t/c///E+pVCqVGhsbSxdddFFp6tSppVKpVFq0aFHpmGOOKa1Zs6ZUKpVK06dPL51zzjnNXvcjH/lI6Yknnij69mnB9pwbm7Ns2bLS4YcfXlqyZEmpVCqVmpqaynOiVNr893jo0KGlBx54oPx45cqVpQ984AOlT3/606V77rmn2bavnXOv2rBhQ+lXv/pVadGiRaV//ud/brbuiSeeKD399NOlUqlUWrduXWnYsGGlO++8c7PvfeXKlc3Gct1115WuuOKKN9x/U1NTaciQIaWf/OQn5ccvvvhis23M79bXXse+P//5z6UPfvCDpeXLl5dKpVLp+uuvL33lK18plUotHxdfZW50fK0591qaQ88+++zrjnmvWrNmTWnAgAGlxx57rFQqbTpOrlixotk2Wzqmsf109M/dhoaG8n9/97vfLf3rv/5reVz9+/cvvfDCC6VSqVS69957Sx//+MebPfe+++4rXXHFFc3m2QMPPFAaOnRoeZvGxsbSvffeW2psbCyVSqXSrbfeWjrjjDNaHP+rPve5z5VuueWW8uPnn3++2frTTjuttGDBgs0+l+3rtd+n+fPnlw477LDS448/vl0/h9/oe765ObelMbLjaY15tC0uu+yy0q233vqG27322NLQ0FB66KGHSj/96U+bHd9KpVJp5syZpbPOOqu0fv360oYNG0pnn312ad68eaVSqVR6/PHHS/X19eXX+OhHP1p66KGHys9t6TO+o3ImUwc2a9asfOELX0ifPn1y3333veH29fX12XPPPVNRUZEkeec735m99957i9sfeeSR2X333V+3vGfPnjnggAOSJJWVlTnooIOyZMmSJElFRUU2bNhQ/tfXl156KXvttdc2vzeKae258eKLL6Zz586pqalJsun7/uqc2Fpz587Nhz/84QwfPjyzZs3aqud07tw5Rx11VHbbbbfXrevdu3f222+/JElVVVUOOOCA8rx8rZqammb/YnDIIYdscdt/9Otf/zq77LJLPvrRjybZ9L439/8Irau9jn1PPvlk+vbtm+7duydJPvzhD2fu3LlJWj4u8tbRmnPvzc6hefPm5bDDDsshhxySZNNx8p3vfOdWviO2l47+ufuPn6urV69OZeWmHxFKpVJKpVL5bJHX/r1u5cqVmTp1aq644ooW911ZWZljjz22/Lpb+7n7zDPP5Mknn8zIkSPLy971rne94fNoG8cdd1yGDRuWm2++ucXttmW+v9H3fGvnHB3H9phH69evz4QJE3LyySdn8ODBufTSS8vHrWXLlmXkyJEZOHBgzj333GZnAq9evTpXXXVVTj755Jx44om59tpr09jYuNn977bbbqmrq8tOO+30unV//OMfc+SRR6ZLly7p3Llz/uVf/qX8d8SDDz44e+65Z/k1evXqleeee24bv0odS+f2HgBvzh//+MesWrUqRxxxRF544YXMmjUrH//4x5Ns+kF4yJAh5W1PO+20nHLKKTnjjDNy4YUXZt68eenfv38GDBiQI488stA41q5dm1mzZuWSSy5Jkuy///4588wzM2DAgOy2226prq7eqtMR2X7aYm7sv//+Oeigg3LMMcfk8MMPz6GHHpohQ4Y0+8HmwgsvTNeuXcuPn3nmmWavMWvWrFx22WU55JBDMm7cuCxbtqx8AE42naLa2NiYD33oQ7ngggtSVVW11V+D5cuXZ/78+eXTW1vS1NSU2267LQMGDGi2fHP7//Of/5yamppceOGF+dvf/pb3vOc9ueKKK1JbW7vVY6OY9jz27b///vn973+fZ599Nj179sy8efPy8ssvZ9WqVeUf/JLXHxd5a2jLube5ObRmzZry6fkDBw7M2WefnYqKivz5z39O586dc+655+b555/PgQcemMsuuyzdunXbzl8BtuSt8rl77rnnZtGiRXnnO99Z/mGve/fuGTt2bIYOHZrq6uo0NTXl1ltvLT9n7NixufDCCzf7jz8tmTFjxus+d0eNGpVSqZTDDjssl1xySaqrq/PnP/85e+65Z6666qr84Q9/yB577JEvf/nLef/7379N+6P1HHzwwVmwYEGOOeaY7TLf3+h7/mbnHDu2ovPo29/+dnbbbbfccccdSZJJkyblpptuysUXX5xrr702H/jAB3L++efn2WefzeDBg3P00UcnScaPH58PfOADGTduXJqamjJq1KjMmjUrn/70p7dp/AceeGBmz56d4cOHJ9l0iVxDQ8Prtnvqqafy+OOP56tf/eqb+jp1FCJTB3XHHXdkyJAhqaioyHHHHZdrr702y5YtS5IcddRRmTJlyuuec+SRR+anP/1pHnzwwTzyyCP54he/mLPPPjuf/exn39QYNm7cmIsvvjhHHHFEjj322CTJc889l/vuuy8//vGP8+53vzvf/va3c/nll+fGG29882+WbdIWc6OysjLTpk3Lk08+mYceeij33ntvbr755sydO7f8w/aUKVPSu3fv8nP+8drlRYsWpaGhIUcccUR5nHPmzMnnPve5JMnPfvaz1NbWZvXq1bn00ktz/fXX5+KLL96q97969eqcd955Oeuss7bqX3m/9rWvZeedd85pp51WXral/Tc1NeWBBx7I7bffnl69euWWW27JZZddlv/4j//YqrFRXHse+9773vfm6quvzsUXX5yKioryca9z5//3Ubq54yJvDW019zY3h9797nfn5z//eXbfffcsX7485513Xrp165ZTTjmlfFyaOXNm9thjj4wfPz7XXXddxo8f3zpfCF7nrfC5myTf+ta30tTUlBtvvDH//u//nmuuuSarV6/OjBkzcscdd+R973tf7r777px//vn54Q9/mB/96Efp0qVLjjnmmG36en3rW9/KU089lenTp5eXzZgxI7W1tVm/fn3GjRuXsWPHZvLkyWlqaspvf/vbfOlLX0pdXV1+/OMf57zzzsu99967Tfuk9ZT+//vOJNtnvrf0Pb/77rvf1Jxjx1d0Hi1YsCCrV6/O/Pnzk2w6s2n//fdPsukecVdffXWSZJ999mkWOBcsWJDf/e53ueWWW5Js+keef4zvW+ukk07Ks88+m8985jPZddddc9BBB+WBBx5ots3zzz+fL3zhCxkzZsyb2kdHIjJ1QOvXr8+8efNSVVWVu+66K0myYcOGzJ49+w0n7K677ppjjz02xx57bPr165d///d/f1ORqbGxMaNGjUq3bt3K/9Mmm25e1rt377z73e9Oknzyk5/c7M36aB1tPTd69+6d3r17l284+5vf/CbHHXfcG45z1qxZaWhoKP8AtX79+uyyyy7lv+y+embQrrvumlNOOaV84H8jr7zySj7/+c/nX/7lX3LWWWe94fYTJkzIX//619xwww3lU/hb2n9tbW0OPPDA9OrVK0kyePDgzX4I0jp2hGPfJz7xiXziE59Ikvzud7/L97///ey6665JtnxcpONrq7m3pTlUVVVVvoRz9913z4knnphHH300p5xySmpra3P44YeXP3dPPPHEXHnlldvjbbMV3iqfu6+qrKzMySefnOOOOy7XXHNNfvnLX2a33XbL+973viSbzqK74oorsnLlyvzmN7/JAw880OyMpEGDBuVb3/rWFsdx6623Zt68eZk+fXqzS05e/dytqqrK8OHDc95555WX19bWpq6uLsmmy2ouvfTSrFixonzpMu3r97///VadWba1872l73lLc+5//a//tf3eFG2u6DwqlUoZM2bMNp+pXiqVMm3atOyzzz5vduhJNh07L7744vI/in/rW98q/7yQbLrK4swzz8w555xTPtP1rcw9mTqg++67L+9973tz//33Z8GCBVmwYEG+853v5M4772zxeb/61a+yevXqJJv+h1q0aFF69uy5zftvamrK5Zdfnk6dOmXcuHHl62KTTfeUeOSRR/Lyyy8nSX7+8587pbkNtdXcWLZsWR577LHy4/r6+qxYsWKr5tOrfyGfNWtWeYy//OUvkyQPP/xw/v73v5fv6bVx48bMnz8/ffv2fcPXXbduXT7/+c/n4IMPzkUXXfSG23/961/PwoULc/311ze7FK+l/X/oQx9KfX19nn/++STJL37xi/Tp0+cN98X20d7HviR54YUXkmyab1OmTCnHzJaOi3R8bTH3WppDy5cvL/82sFdeeSULFiwo/wvtxz/+8fzud78r7+f+++93XGpDb4XP3RUrVjT7Dav33HNPeQ717NkzixYtyvLly5MkDzzwQHbddde8853vzDXXXNPsfSeb7hG2pR/2Z86cmdtvvz233HJLs0uMX3755bz00kvlr8Xdd99d/tzt169fdt555/JvKn7ooYfSrVs39x3bQdx777257bbb3vAf9rZlvrf0Pd/WOUfHsD3m0YABA/Ld7363/Hf41atX56mnnkqSHHHEEeX70D377LPNfiPdgAEDctNNN5Xvw7RixYo8++yz2/we1q1bVz6OLVmyJLfddlvOPPPMJJvuI3bmmWdmxIgROeWUU7b5tTsiZzJ1QK/+Cu1/1L9//zQ1NWXJkiWvu461X79+GTduXJ544olcd9115dMR991334wePXqL+zn//PPzu9/9LklywgknpHfv3rn55ptz//3354c//GF69+5dPhX70EMPzZgxY3Lcccflt7/9bU466aRUVVWlurraKfttqK3mxsaNG/PNb34zzz33XN7xjnekqakpX/ziF7fq8rR7770373nPe7Lvvvs2W37iiSeWr4EePXp0KioqsnHjxvTv379ZNPrUpz6VZcuWpaGhIR/60Idy9NFHZ9y4cbnjjjvym9/8JqtWrSr/5fmEE04o/2voP/rTn/6UG2+8Mfvtt1+GDRuWZNNfpK+//vr85S9/2eL+d95551x99dU599xzUyqVUlNTk+uuu+4N3zPbR3sf+5JNvzJ5yZIl2bBhQwYOHJgzzjgjSVo8LtLxtcXca2kOPfLII5kyZUoqKyuzcePGHHPMMeVLfPfee++ce+65GTZsWCoqKtKzZ8987Wtfa40vA5vxVvjc/d//+3/niiuuKIfMHj16ZNKkSeXxnnPOOTnttNPSpUuXVFVV5Rvf+MY2h/TVq1fnmmuuyd57713+wauqqio/+MEPsnz58lxwwQVpbGxMU1NTevXqVT52VlRU5N/+7d9yxRVXZP369dlpp50ydepUIb8dXXjhhamqqsorr7ySXr165aabbsrBBx+cp556arvMd9/zt4ftPY8++9nPZurUqTn55JNTUVGRioqKnH/++enVq1euuuqqfPnLX868efPSs2fPZr/858orr8ykSZPKlzx36dIlV1555WbPbGpsbMxHPvKRrF+/PqtXr86HPvShnHLKKbngggvy0ksv5fTTTy9fGTFq1KgceOCBSZKbbropzzzzTP7zP/8z//mf/5kkOeOMM/KpT32qdb64O4CK0j9eAMnb1je/+c28/PLLueyyy1p9XwMGDMgNN9zQ7L4B0Jpmz56dn/3sZ21yaZv53bE49tFe2nLu9enTJ48++mh22WWXVt8XJJvugTJhwoTMnj271fd1+umn56yzzspHPvKRVt8X29/2Phb6rGV7aatjy+LFi/OpT30qDz74YKvupy25XI4km87Q+MlPfpILL7yw1faxePHiDBkyJBs2bGh2o1xobe94xzuycOHC8llLrcH87pgc+2gvbTH3Hn300QwZMiR77LFHs/vOQWvr0qVLli9fniFDhmz2NyxtD+vXr8+QIUPy7LPPNvutenQs2+tY6LOW7a1bt26ZOHHiVv226jfrnnvuyXnnnZc99tij1fbRHpzJBGzR1KlT85Of/OR1y7/zne+Ub0ILAGwfPncB6OhEJgAAAAAKc+40AAAAAIWJTAAAAAAUJjIBAAAAUJjIBADwJsydOzcnnXRS+vfvnw9+8IM555xz8vDDD+eb3/xmRo0aVd6uT58+OeSQQ9K/f/8cffTRGT9+fBobG5Ns+nXbv/71r5u97uzZs/OZz3ym/HjAgAE56KCD0r9///KfsWPHlrft27dv+vfvn0MPPTSDBw/OT3/60zZ49wAAr+f3OwIAbKNbbrklN910U7761a/mgx/8YLp06ZJf/OIXue+++7Lzzju/bvu77ror++67b5566qmcccYZ2W+//ZqFpDdyww035KijjtrsukMOOSS33XZbmpqacvvtt+eSSy7Jz3/+81RXV7/p9wcA8GY4kwkAYBu89NJLmTJlSkaPHp3jjjsuO++8c7p06ZIBAwbksssua/G5vXr1ymGHHZY//elP231clZWVGTJkSF5++eU888wz2/31AQDeiMgEALANHnvssaxbty4f+9jHtvm5f/7zn/PII4+kb9++231cjY2NmT17drp06ZIePXps99cHAHgjLpcDANgGq1atyjvf+c507rz1f40aOnRoOnXqlG7duuXkk0/Opz71qfK6f/3Xf02nTp3Kjzds2JADDjig2fNfu82Xv/zlfPrTn06S/Pa3v01dXV1eeeWVdOrUKRMnTszuu+/+Zt8eAMCbJjIBAGyDmpqarFy5Mhs3btzq0HTnnXdm33333ey666+/vtn9lmbPnp0f/OAHLW7zjw4++ODcdtttWbNmTa666qo88sgjGThw4Fa+GwCA7cflcgAA26B///6pqqrKvffe295DaWaXXXbJNddck7vuuiuLFi1q7+EAAG9DIhMAwDbYbbfdcuGFF2bs2LG5995788orr2TDhg35+c9/nokTJ7br2GpqanLKKafk+uuvb9dxAABvTy6XAwDYRmeddVb22GOPTJs2LaNGjcouu+ySAw88MJ///Ofzq1/9arvv7/Of/3yzezIdddRRWwxJI0eOzEc/+tH88Y9/zP7777/dxwIAsCUVpVKp1N6DAAAAAKBjc7kcAAAAAIWJTAAAAAAUJjIBAAAAUJjIBAAAAEBhIhMAAAAAhYlMAAAAABQmMgEAAABQmMgEAAAAQGEiEwAAAACF/X+edPeER6Hg5AAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# ciphers  = info_df.CIPHER\n",
    "# time_sec = info_df['TIME(SEC)']\n",
    "sns.barplot(x=\"CIPHER\",y=\"TIME(SEC)\",data=info_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 485,
   "id": "surrounded-subscriber",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='CIPHER', ylabel='TASA(BITS/SEC)'>"
      ]
     },
     "execution_count": 485,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABJYAAAJeCAYAAADm/I+QAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAABD7ElEQVR4nO3de5RWZd0//jcMkCdGxBMQpkmPSLpUElN7FBVUlNOIiZqY5IHSPJSpKUkcTFPSb2ul6GNSEglpIqbJl9RQs7QHMzPFSCnTlGRAOTQCKjDcvz/8Oj8nkIEtc8+Ar9darDV77+ve+7Pnvtj3zHuufe0WpVKpFAAAAADYQC2bugAAAAAANk2CJQAAAAAKESwBAAAAUIhgCQAAAIBCBEsAAAAAFCJYAgAAAKCQzS5YGjt2bHr16pWuXbtmzpw56/Wau+66KwMGDEhVVVWOP/74/PGPf2zkKgEAAAA2fZtdsNS7d+9Mnjw5H//4x9er/eLFi/Pd7343EyZMyL333ptzzz03I0eObOQqAQAAADZ9rZq6gI2tR48ea13/zDPP5LrrrsuyZcuSJBdccEEOP/zwlEqllEqlLFu2LDvssEPefPPNdOjQoZwlAwAAAGySNrtgaW1qamoyatSo3HLLLdlpp52yYMGCnHDCCZk2bVrat2+fK664IoMGDUplZWVWr16d2267ralLBgAAAGj2PhLB0tNPP525c+dm2LBhdetatGiRf/7zn/nkJz+ZyZMn56677sruu++e6dOn57zzzssvf/nLtGjRogmrBgAAAGjePhLBUqlUSteuXTN58uQ1tt1///1p27Ztdt999yRJ3759M3z48CxevDjt27cvd6kAAAAAm4zNbvLutenevXv++c9/ZubMmXXrnn322ZRKpXTu3DmzZ8/OwoULkyQzZ87MNttsk+22266pygUAAADYJLQolUqlpi5iY7ryyivz4IMP5o033sh2222Xdu3a5f/+3/+bZ599Ntdee23+/e9/Z+XKldlll11y8803p2XLlpkwYULuvPPOtG7dOm3atMlll132gZOAAwAAAPCuzS5YAgAAAKA8PhK3wgEAAACw8W02k3e//fbbee6557LjjjumoqKiqcsBAAAA2OTV1tbm9ddfz957750ttthije2bTbD03HPPZciQIU1dBgAAAMBmZ/LkyWudj3qzCZZ23HHHJO+e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AAAAAIUIlgAAAAAoRLAEAAAAQCGCJQAAAAAKESwBAAAAUIhgCQAAAIBCBEsAAAAAFCJYAgAAAKAQwRIAAAAAhQiWAAAAAChEsAQAAABAIYIlAAAAAAoRLAEAAABQiGAJAAAAgEIESwAAAAAUIlgCAAAAoBDBEgAAAACFCJYAAAAAKESwBAAAAEAhgiUAAAAAChEsAQAAAFCIYAkAAACAQlo11GDRokW5995785vf/CbPP/98li5dmm222SZ77rlnevbsmUGDBqV9+/blqBUAAACAZmSdwdJ1112X++67L4cddlhOOOGEdOnSJVtvvXWWLVuWF198MU8++WQGDRqUAQMG5OKLLy5XzQAAAAA0A+sMljp06JBf//rXadOmzRrbPv3pT2fAgAF55513MmXKlEYrEAAAAIDmaZ1zLJ166qlrDZXe72Mf+1hOPfXUjVoUAADA5mjVypVNXQJNyPvP5qjBOZaeeuqpPPzww7nkkkvW2HbdddflyCOPzH777dcYtQEAAGxWWrVune8P/0pTl0ET+cbVP2zqEmCja/CpcD/84Q9zwAEHrHXbZz/72dx8880bvSgAAAAAmr8Gg6W//vWvOfTQQ9e67XOf+1yee+65Bg8yduzY9OrVK127ds2cOXPW2qa2tjZjxozJkUcemaOOOsq8TQAAAADNXIPB0tKlS7PyA+4DXbVqVZYtW9bgQXr37p3Jkyfn4x//+Ae2ue+++/LKK6/kwQcfzM9//vPccMMNmTt3boP7BgAAAKBpNBgs7b777nnsscfWuu2xxx7L7rvv3uBBevTokY4dO66zzfTp0zN48OC0bNky7du3z5FHHpn777+/wX0DAAAA0DQanLz7S1/6UkaNGpXVq1fnyCOPTMuWLbN69erMmDEjV1xxRS677LKNUsi8efPSqVOnuuWOHTumurp6rW1rampSU1NTb90HtQUAAACgcTQYLA0YMCBvvPFGLr300qxcuTLt2rXLkiVL0rp161xwwQXp379/OeqsZ+LEiRk3blzZjwsAAADA/6/BYClJTj/99AwePDhPP/10lixZknbt2qV79+7ZZpttNlohHTt2zGuvvZZ99tknyZojmN5v6NChGTRoUL111dXVGTJkyEarBwAAAIB1azBYev3117Pjjjtmm222WevT4Z577rnsvffeH7qQY445JlOmTMnRRx+dJUuWZMaMGZk8efJa21ZWVqaysvJDHxMAAACA4hqcvLtPnz71lo8++uh6y6eddlqDB7nyyivTs2fPVFdX5/TTT0+/fv2SJMOGDcusWbOSJFVVVencuXOOPvronHjiiTn33HOzyy67rPeJAAAAAFBeDY5YKpVK9ZYXL168zu1rM2LEiIwYMWKN9ePHj6/7uqKiImPGjGlwXwAAAAA0Dw2OWGrRosUGLQMAAADw0dBgsAQAAAAAa9PgrXBvv/12vaetLVu2rG65VCrlnXfeabzqAAAAAGi2GgyWrrrqqnrLJ5xwQr3lwYMHb9yKAAAAANgkNBgsDRo0qBx1AAAAALCJaXCOpeeeey5z5sypW160aFEuuuiiDBw4MCNHjsyyZcsatUAAAAAAmqcGg6Xvfve7eeONN+qWL7/88rz88ss56aST8re//S3XXnttoxYIAAAAQPPUYLD04osvpkePHkmSmpqa/O53v8t1112XIUOG5Pvf/34eeeSRRi8SAAAAgOanwWCptrY2rVu3TpL8+c9/zg477JBPfvKTSZKOHTumpqamcSsEAAAAoFlqMFj61Kc+lV/96ldJkunTp+fggw+u2zZ//vy0bdu28aoDAAAAoNlq8KlwF198cc4555yMHj06LVu2zM9+9rO6bdOnT89nPvOZRi0QAAAAgOapwWCpR48eeeSRR/Lyyy9nt912yzbbbFO37bDDDkvfvn0btUAAAAAAmqcGg6XDDjssPXv2zGGHHZbdd9+93rb/XAYAAADgo6PBOZamTJmSffbZJ/fee2969eqV008/PT/5yU/yj3/8oxz1AQAAANBMNThiaaeddsrgwYMzePDgrFq1Kk8++WR++9vf5rzzzsvKlSvrRjQddNBBadOmTTlqBgAAAKAZaDBYqte4VascfPDBOfjgg3PppZdm7ty5efTRRzNp0qT87W9/y5lnntlYdQIAAADQzGxQsPSef/zjH3nxxRfTrVu3DBkyJEOGDNnYdQEAAADQzDU4x9LVV1+de++9t275nnvuSf/+/fPtb387ffv2zaOPPtqoBQIAAADQPDUYLM2YMSMHHHBA3fL3v//9XH755Zk5c2bGjBmTG2+8sVELBAAAAKB5ajBYWrx4cTp16pQkmTNnTpYsWZLBgwcnSQYOHJiXX365UQsEAAAAoHlqMFhq27Zt3njjjSTJH//4x+y99951T39btWpVSqVS41YIAAAAQLPU4OTdxx57bC688MIcddRRmTBhQoYNG1a37Zlnnskuu+zSqAUCAAAA0Dw1OGLpoosuyoEHHpjf//73OfHEE/OFL3yhbttf//rXnHzyyY1aIAAAAADNU4MjliZMmJDzzjtvrduGDh260QsCAAAAYNPQ4Iilm2++uRx1AAAAALCJaTBYMjk3AAAAAGvT4K1wtbW1mTp16joDphNOOGGjFgUAAABA89dgsLRq1arcc889H7i9RYsWgiUAAACAj6AGg6Utttgit912WzlqAQAAAGAT0uAcSwAAAACwNg0GS506dSpHHQAAAABsYhoMlqZNm1aOOgAAAADYxDQ4x9Jhhx2WFi1arLPNb37zm41VDwAAAACbiAaDpWuvvbYcdQAAAACwiWkwWPrsZz9bjjoAAAAA2MSs11PhFi1alD//+c9ZunRpkuTOO+/M2Wefne9///t55513GrVAAAAAAJqnBkcsPfjgg7nkkkuy9dZbZ9WqVfnKV76Se+65J4ceemgeeeSRvPnmmxk1alQ5agUAAACgGWkwWPrBD36QG2+8MYccckh+85vf5Nxzz82vf/3rdOrUKaeddloGDx4sWAIAAAD4CGrwVrh58+blkEMOSZIcfvjhadOmTTp16pQk6dChQ5YtW9a4FQIAAADQLK3XHEvv17p168aoAwAAAIBNTIO3wq1YsSLf/OY365aXL19et1wqlbJixYrGqw4AAACAZqvBYOnss8/eoGUAAAAAPhoaDJbOO++8ctQBAAAAwCZmnXMsPf/88+u1k/VtBwAAAMDmY50jlsaMGZNtttkmVVVVOeCAA7LzzjvXbVuwYEGefPLJ3HPPPVm2bFl+9rOfNXqxAAAAADQf6wyWbr/99jzyyCO54447cvnll6dly5bZeuuts2zZsiTJwQcfnFNPPTWHHXZYWYoFAAAAoPlocI6lI444IkcccURWrlyZf/7zn6mpqcm2226bT3ziE2ndunU5agQAAACgGWowWHpP69at86lPfareun//+9+ZNm1ahgwZstELAwAAAKB5W+fk3WtTW1ubhx56KOeff34OOeSQ3HHHHY1RFwAAAADN3HqPWPrLX/6SX/ziF5k+fXrefvvtrFixItdff3169erVmPUBAAAA0Ew1OGLpRz/6UQYMGJCTTz45c+fOzeWXX57HH3887dq1y7777luOGgEAAABohhocsXTdddelXbt2GTt2bI499ti0aNGiHHUBAAAA0Mw1OGJp4sSJOeKIIzJixIj07Nkz11xzTZ577rkNPtBLL72Uk046KX369MlJJ52Ul19+eY02CxcuzJe//OUMGDAgxx57bEaPHp1Vq1Zt8LEAAAAAaHwNBksHHnhgrr766jz++OO56KKL8sILL+TEE0/MwoULc8cdd2Tx4sXrdaBRo0bllFNOyQMPPJBTTjklI0eOXKPNzTffnC5duuS+++7LL3/5y/zlL3/Jgw8+uOFnBQAAAECjW++nwm255ZY57rjjMmHChDz00EP52te+lvvuuy+HH354g69duHBhZs+enf79+ydJ+vfvn9mzZ2fRokX12rVo0SLLli3L6tWrs2LFiqxcuTI777zzhp0RAAAAAGWx3k+Fe7+OHTvm7LPPztlnn51nnnmmwfbz5s3LzjvvnIqKiiRJRUVFdtppp8ybNy/t27eva/fVr341559/fg455JC89dZbGTJkSPbff/819ldTU5Oampp666qrq4ucCgAAAAAFNRgsLV++PEmy1VZbJUlKpVKmTJmSOXPmpHv37unXr99GK+b+++9P165dM3HixCxbtizDhg3L/fffn2OOOaZeu4kTJ2bcuHEb7bgAAAAAbLgGb4W78MIL681zNHbs2Pyf//N/smDBglx55ZW59dZbGzxIx44dM3/+/NTW1iZJamtrs2DBgnTs2LFeu0mTJmXgwIFp2bJl2rZtm169euWJJ55YY39Dhw7NQw89VO/f5MmTG6wDAAAAgI2nwWDpL3/5S3r16pUkWbFiRe6888784Ac/yPXXX58f/vCHufPOOxs8yPbbb59u3bpl2rRpSZJp06alW7du9W6DS5LOnTvnt7/9bd2x/vd//zf/9V//tcb+Kisr07lz53r/OnTo0PDZAgAAALDRNBgsvfXWW6msrEySPPfcc2nVqlUOOuigJMk+++yT119/fb0ONHr06EyaNCl9+vTJpEmTMmbMmCTJsGHDMmvWrCTJt771rTz11FMZMGBAjjvuuOy222458cQTC50YAAAAAI2rwTmWdtpppzz//PPZc8898/jjj9ebTLumpiZt2rRZrwN16dIlU6ZMWWP9+PHj677+xCc+kQkTJqzX/gAAAABoWg0GS2eccUbOPPPMdO/ePY899lhuuOGGum2PPfZYunbt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mH3jg/zqV7/Kcccdl6qqqhxzzDG56KKL6rb16tVrjff4+OOPzxNPPFFv3de//vUcdNBBWblyZb31Xbt2raujqqoqL7zwQpJkxYoVOfPMM3PggQfmwAMPrPeaP/3pTzn55JPTt2/f9O3bN2PHjs17U8a9d+4nn3xykmTx4sUZNmxY+vTpkwEDBuS8887LokWLGjx+8m4/fu+1ffv2rXu6pf5dPk117UuSqVOnZsCAAenXr1/OPvvsLFmyJMm6r4v6xualMftfQ/3o05/+dL1jLF68uK6Gv/71rxkyZEjdNfDRRx9N8u61saqqaq39mY1nU/7cXb16dU466aQMHDgwAwcOzJlnnpm5c+fWbb/rrrvqajz++OPzxz/+cY3ahg8fXq+PPfHEE9l3331TVVWVmpqarF69Oueff3769OmTgQMH5vTTT88rr7xSr/5jjjmm7vv0u9/9rm7bkiVL8o1vfCN9+vRJv379Mm7cuCTv/kxQVVWV7t2755FHHlnn946N4733aeDAgTnqqKNyzjnn5E9/+lOSjfs5/EHv+fv9Z5/zWbvpaIx+tCEuu+yyTJo0qcF2X/ziF9O7d+/ccsstSZL58+fni1/8Yvbff/8cf/zx9drW1tbmqquuSv/+/dOnT596v4fMmDEjxx9/fPr3759+/frl1ltvrXvd/fffn759+67xe81mocQmY9KkSaVTTz21dNBBB5XeeeedUqlUKk2dOrV0/vnnr7X96NGjS9dcc01p9erVpVKpVFq0aFHpX//6V6lUKpWuv/760jXXXFOv/e9///vSG2+8Udpjjz1KS5curVv/6quvlv7yl7+USqVSqba2tvS1r32tNG7cuFKpVCrNnj27dPjhh5eWLVtWKpVKpYkTJ5bOOuusevs94ogjSi+88MKHPX3WYWP2jbWZP39+6cADDyy99tprpVKpVFq9enVdnyiV1v4eDxo0qDRz5sy65cWLF5cOOOCA0oknnli6//7767X9zz73npUrV5Yef/zx0uzZs0uf/exn62174YUXSi+99FKpVCqV3nnnndLJJ59c+sUvfrHWc1+8eHG9Wq655prS8OHDGzz+6tWrS1VVVaVf//rXdctvvPFGvTb6d+Nrqmvf3//+99IhhxxSWrhwYalUKpVuvPHG0re//e1SqbTu6+J79I3NQ2P2v3X1o1dffXWN6957li1bVurVq1fp6aefLpVK714rFy1aVK/NB13X2Dg29c/dmpqauq9/8pOflM4999y6urp37156/fXXS6VSqTRjxozSscceW++1Dz30UGn48OH1+tjMmTNLgwYNqmtTW1tbmjFjRqm2trZUKpVKt912W+m0005bZ/3v+cpXvlKaMGFC3fKCBQvqbT/11FNLDz/88Fpfy8b1n+/TAw88UNp///1Lf/7znzfq53BD7/na+twH1Ujz0xj9aENceumlpdtuu63Bdv95bampqSk9+eSTpUceeaTe9a1UKpXuuOOO0hlnnFFasWJFaeXKlaUzzzyzNG3atFKpVCr9+c9/LlVXV9ft48gjjyw9+eSTda9d1+f7psyIpU3I1KlT89WvfjVdu3bNQw891GD76urq7LzzzmnRokWSZLvttkunTp0+sP3BBx+c7bfffo31nTt3zqc//ekkScuWLbPPPvvktddeS5K0aNEiK1eurPsL65tvvpkOHTps8Lnx4TR233jjjTfSqlWrtGvXLsm77/t7fWJ93XfffTnssMNyyimnZOrUqev1mlatWuVzn/tc2rZtu8a2PfbYI7vttluSpE2bNvn0pz9d1y//U7t27er9ZWC//fb7wLbv9/vf/z5bb711jjzyyCTvnvfa/o/QuJrq2jdnzpx069Yt7du3T5Icdthhue+++5Ks+7rI5qUx+1/RfjRt2rTsv//+2W+//ZK8e63cbrvt1vOM2Bg29c/d93+uLl26NC1bvvsrQalUSqlUqhsV8p8/1y1evDjjxo3L8OHD13nsli1bpnfv3nX7Xd/P3Zdffjlz5szJ0KFD69btuOOODb6O8jj66KNz8skn58c//vE6221If2/oPV/fPsemY2P0oxUrVmTs2LE54YQTMnDgwFxyySV116358+dn6NCh6du3b4YNG1ZvtO/SpUtz+eWX54QTTsiAAQNy5ZVXpra2dq3Hb9u2bXr06JEtt9xyjW3PP/98Dj744LRu3TqtWrXKf//3f9f9jLjvvvtm5513rttHly5d8q9//WsDv0ubnlZNXQDr5/nnn8+SJUty0EEH5fXXX8/UqVNz7LHHJnn3l9+qqqq6tqeeemoGDx6c0047LRdccEGmTZuW7t27p1evXjn44IM/VB1vv/12pk6dmm984xtJkj333DOnn356evXqlbZt26aysnK9hhqy8ZSjb+y5557ZZ599cvjhh+fAAw/MZz7zmVRVVdX7ReaCCy7Ixz72sbrll19+ud4+pk6dmksvvTT77bdfrrrqqsyfP7/uopu8O/y0trY2PXv2zPnnn582bdqs9/dg4cKFeeCBB+qGrq7L6tWrc/vtt6dXr1711q/t+H//+9/Trl27XHDBBXnllVfyiU98IsOHD0/Hjh3XuzY+nKa89u25556ZNWtWXn311XTu3DnTpk3L8uXLs2TJkrpf9pI1r4tsPsrZ/9bWj5YtW1Y3/L5v374588wz06JFi/z9739Pq1atMmzYsCxYsCB77bVXLr300my77bYb+TvA2mwun7vDhg3L7Nmzs91229X9gte+fftcccUVGTRoUCorK7N69ercdtttda+54oorcsEFF6z1Dz7rMnny5DU+dy+++OKUSqXsv//++cY3vpHKysr8/e9/z84775zLL788f/3rX7PDDjvkm9/8Zv7rv/5rg45H49l3333z8MMP5/DDD98o/b2h97xon6N5+7D96Ec/+lHatm2bu+66K0ly7bXX5pZbbsmFF16YK6+8MgcccEDOO++8vPrqqxk4cGAOPfTQJMnVV1+dAw44IFdddVVWr16diy++OFOnTs2JJ564QfXvtddeufvuu3PKKackeff2t5qamjXavfjii/nzn/+cMWPGFPo+bUoES5uIu+66K1VVVWnRokWOPvroXHnllZk/f36S5HOf+1yuv/76NV5z8MEH55FHHskTTzyRp556Kl//+tdz5pln5stf/nKhGlatWpULL7wwBx10UHr37p0k+de//pWHHnooDz74YHbaaaf86Ec/ymWXXZYf/vCHxU+WDVKOvtGyZcvcdNNNmTNnTp588snMmDEjP/7xj3PffffV/YJ9/fXXZ4899qh7zfvvRZ49e3Zqampy0EEH1dV5zz335Ctf+UqS5De/+U06duyYpUuX5pJLLsmNN96YCy+8cL3Of+nSpTnnnHNyxhlnrNdfc7/zne9kq622yqmnnlq37oOOv3r16sycOTN33nlnunTpkgkTJuTSSy/NT3/60/WqjQ+vKa99n/zkJzNixIhceOGFadGiRd11r1Wr//+jc23XRTYf5ep/a+tHO+20Ux599NFsv/32WbhwYc4555xsu+22GTx4cN216Y477sgOO+yQq6++Otdcc02uvvrqxvlGUM/m8LmbJOPHj8/q1avzwx/+MP/zP/+T0aNHZ+nSpZk8eXLuuuuu7L777pk+fXrOO++8/PKXv8yvfvWrtG7dOocffvgGfb/Gjx+fF198MRMnTqxbN3ny5HTs2DErVqzIVVddlSuuuCLXXXddVq9enWeeeSYXXXRRevTokQcffDDnnHNOZsyYsUHHpPGU/t88MsnG6e/res+nT59eqM/R/H3YfvTwww9n6dKleeCBB5K8O4Jpzz33TPLunG8jRoxIkuyyyy71Qs2HH344zz77bCZMmJDk3T/qvD9wX1/HH398Xn311XzhC1/INttsk3322SczZ86s12bBggX56le/mlGjRhU6xqZGsLQJWLFiRaZNm5Y2bdrk3nvvTZKsXLkyd999d4OddJtttknv3r3Tu3fv7L333vmf//mfQsFSbW1tLr744my77bZ1/1GTdycg22OPPbLTTjslSY477ri1TrhH4yh339hjjz2yxx571E0Y+4c//CFHH310g3VOnTo1NTU1db8wrVixIltvvXXdD7jvjQDaZpttMnjw4LqLfUPeeuutnH322fnv//7vnHHGGQ22Hzt2bP75z3/m5ptvrhuev67jd+zYMXvttVe6dOmSJBk4cOBaP/hoHM3h2tevX7/069cvSfLss8/mZz/7WbbZZpskH3xdZPNQrv73Qf2oTZs2dbdobr/99hkwYED+9Kc/ZfDgwenYsWMOPPDAus/eAQMG5Fvf+tbGOG0asLl87r6nZcuWOeGEE3L00Udn9OjReeyxx9K2bdvsvvvuSd4dKTd8+PAsXrw4f/jDHzJz5sx6I4/69++f8ePHf2Adt912W6ZNm5aJEyfWu53kvc/dNm3a5JRTTsk555xTt75jx47p0aNHkndvmbnkkkuyaNGiutuSaVqzZs1arxFk69vf1/Wer6vPfepTn9p4J0XZfdh+VCqVMmrUqA0ekV4qlXLTTTdll112KVp6knevnRdeeGHdH8LHjx9f9/tC8u7dFKeffnrOOuusuhGtmztzLG0CHnrooXzyk5/Mb3/72zz88MN5+OGHc+utt+YXv/jFOl/3+OOPZ+nSpUne/U80e/bsdO7ceYOPv3r16lx22WWpqKjIVVddVXefa/Lu/BBPPfVUli9fniR59NFHDVcuo3L1jfnz5+fpp5+uW66urs6iRYvWqz+990P41KlT62p87LHHkiR//OMf8+9//7tujq5Vq1blgQceSLdu3Rrc7zvvvJOzzz47++67b772ta812P773/9+nnvuudx44431brNb1/F79uyZ6urqLFiwIEnyu9/9Ll27dm3wWGwcTX3tS5LXX389ybv97frrr68LMNd1XWTzUI7+t65+tHDhwroneb311lt5+OGH6/4ae+yxx+bZZ5+tO85vf/tb16Yy2Rw+dxctWlTvyaj3339/Xf/p3LlzZs+enYULFyZJZs6cmW222SbbbbddRo8eXe+8k3fn+/qgX/DvuOOO3HnnnZkwYUK924eXL1+eN998s+57MX369LrP3b333jtbbbVV3ROGn3zyyWy77bbmEGsmZsyYkdtvv73BP+ZtSH9f13u+oX2OTcPG6Ee9evXKT37yk7qf4ZcuXZoXX3wxSXLQQQfVzSv36quv1nuSXK9evXLLLbfUzau0aNGivPrqqxt8Du+8807ddey1117L7bffntNPPz3Ju/OCnX766RkyZEgGDx68wfveVBmxtAl473HX79e9e/esXr06r7322hr3pe6999656qqr8sILL+Saa66pG2q46667ZuTIkR94nPPOOy/PPvtskuSYY47JHnvskR//+Mf57W9/m1/+8pfZY4896oZZf+Yzn8moUaNy9NFH55lnnsnxxx+fNm3apLKy0lD8MipX31i1alVuuOGG/Otf/8oWW2yR1atX5+tf//p63Xo2Y8aMfOITn8iuu+5ab/2AAQPq7mkeOXJkWrRokVWrVqV79+71gqLPf/7zmT9/fmpqatKzZ88ceuihueqqq3LXXXflD3/4Q5YsWVL3A/MxxxxT91fP9/vb3/6WH/7wh9ltt91y8sknJ3n3h+cbb7wx//jHPz7w+FtttVVGjBiRYcOGpVQqpV27drnmmmsaPGc2jqa+9iXvPt74tddey8qVK9O3b9+cdtppSbLO6yKbh3L0v3X1o6eeeirXX399WrZsmVWrVuXwww+vu4W3U6dOGTZsWE4++eS0aNEinTt3zne+853G+DbwHzaHz90vfelLGT58eF1w+fGPfzzXXnttXb1nnXVWTj311LRu3Tpt2rTJD37wgw0Oz5cuXZrRo0enU6dOdb9stWnTJlOmTMnChQtz/vnnp7a2NqtXr06XLl3qrp0tWrTId7/73QwfPjwrVqzIlltumXHjxgnvm9AFF1yQNm3a5K233kqXLl1yyy23ZN99982LL764Ufq79/yjYWP3oy9/+csZN25cTjjhhLRo0SItWrTIeeedly5duuTyyy/PN7/5zUybNi2dO3eu9wCfb33rW7n22mvrbmdu3bp1vvWtb611BFNtbW2OOOKIrFixIkuXLk3Pnj0zePDgnH/++XnzzTfzxS9+se4OiIsvvjh77bVXkuSWW27Jyy+/nJ///Of5+c9/niQ57bTT8vnPf75xvrnNRIvS+29w5CPjhhtuyPLly3PppZc2+rF69eqVm2++ud48ANCY7r777vzmN78py21r+vemxbWPplTO/te1a9f86U9/ytZbb93ox4InnngiY8eOzd13393ox/riF7+YM844I0cccUSjH4uNb2NfB33WsrGU69oyd+7cfP7zn88TTzzRqMcpN7fCfURttdVW+fWvf50LLrig0Y4xd+7cVFVVZeXKlfUmu4XGtsUWW+S5556rG53UGPTvTZNrH02pHP3vT3/6U6qqqrLDDjvUm0sOGlPr1q2zcOHCVFVVrfXJSBvDihUrUlVVlVdffbXe0/DYtGys66DPWja2bbfdNt/73vfW6ynTRd1///0555xzssMOOzTaMZqKEUtAnXHjxuXXv/71GutvvfXWuklkAYCNw+cuAJsDwRIAAAAAhRgjDQAAAEAhgiUAAAAAChEsAQAAAFCIYAkAYD3dd999Of7449O9e/cccsghOeuss/LHP/4xN9xwQy6++OK6dl27ds1+++2X7t2759BDD83VV1+d2traJO8+Hvv3v/99vf3efffd+cIXvlC33KtXr+yzzz7p3r173b8rrriirm23bt3SvXv3fOYzn8nAgQPzyCOPlOHsAQDW5NmMAADrYcKECbnlllsyZsyYHHLIIWndunV+97vf5aGHHspWW221Rvt77703u+66a1588cWcdtpp2W233eqFRw25+eab87nPfW6t2/bbb7/cfvvtWb16de6888584xvfyKOPPprKysrC5wcAUIQRSwAADXjzzTdz/fXXZ+TIkTn66KOz1VZbpXXr1unVq1cuvfTSdb62S5cu2X///fO3v/1to9fVsmXLVFVVZfny5Xn55Zc3+v4BABoiWAIAaMDTTz+dd955J0cdddQGv/bvf/97nnrqqXTr1m2j11VbW5u77747rVu3zsc//vGNvn8AgIa4FQ4AoAFLlizJdtttl1at1v9Hp0GDBqWioiLbbrttTjjhhHz+85+v23buueemoqKibnnlypX59Kc/Xe/1/9nmm9/8Zk488cQkyTPPPJMePXrkrbfeSkVFRb73ve9l++23L3p6AACFCZYAABrQrl27LF68OKtWrVrvcOkXv/hFdt1117Vuu/HGG+vNn3T33XdnypQp62zzfvvuu29uv/32LFu2LJdffnmeeuqp9O3bdz3PBgBg43ErHABAA7p37542bdpkxowZTV1KPVtvvXVGjx6de++9N7Nnz27qcgCAjyDBEgBAA9q2bZsLLrggV1xxRWbMmJG33norK1euzKOPPprvfe97TVpbu3btMnjw4Nx4441NWgcA8NHkVjgAgPVwxhlnZIcddshNN92Uiy++OFtvvXX22muvnH322Xn88cc3+vHOPvvsenMsfe5zn/vA8Gjo0KE58sgj8/zzz2fPPffc6LUAAHyQFqVSqdTURQAAAACw6XErHAAAAACFCJYAAAAAKESwBAAAAEAhgiUAAAAAChEsAQAAAFCIYAkAAACAQgRLAAAAABQiWAIAAACgEMESAAAAAIX8f0IBrKn6AhPaAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x=\"CIPHER\",y=\"TASA(BITS/SEC)\",data=info_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 486,
   "id": "peaceful-birmingham",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='CIPHER', ylabel='TASA(MB/SEC)'>"
      ]
     },
     "execution_count": 486,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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fnh+//vfZ968eampqclOO+2Unj17tvZ8AAAAALQDyxWZkqRnz55LjUpz5sz52HMlAQAAAPCvocVzMn3pS19qdnnYsGHNLu++++4rdyIAAAAA2p0WI9OiRYuaXf7LX/7S7HKpVFq5EwEAAADQ7rQYmaqqqgrtBwAAAODTr8XIBAAAAAAtafHE3wsXLsyZZ55ZvvzOO+80u7xw4cLWmQwAAACAdqPFyDR8+PAVugwAAADAv54WI9MJJ5zQFnMAAAAA0I61GJlee+21dOzYMRtvvHGS5N13383VV1+dF154IX379s3RRx+djh07tvqgAAAAAKy6Wjzx9znnnJM///nP5cvnnXdefvnLX6ZXr165/fbbc9lll7XqgAAAAACs+lqMTM8//3y+8pWvJFl80u977rknP/rRjzJixIhcddVV+eUvf9nqQwIAAACwamsxMi1atChrrLFGkuTPf/5z1lxzzWy99dZJkt69e2fu3LmtOyEAAAAAq7wWI1OPHj3y2GOPJUmmTJmSfv36lffNmTMnq6++eutNBwAAAEC7sFy/Xe473/lONt100/ztb3/LDTfcUN734IMP5t///d9bdUAAAAAAVn0tRqbdd989d9xxR/7yl79kyy23zKabblre9/nPfz7bbbdda84HAAAAQDvQYmQaMmRIdt111+yyyy7NAlOS7LDDDq02GAAAAADtR4vnZDrrrLPy3nvv5Zxzzskuu+ySkSNH5v7778/8+fPbYj4AAAAA2oEWj2Tabrvtst122+W73/1u3njjjTz88MOZOHFivv/972eLLbbILrvskp133jm9e/dui3kBAAAAWAW1GJk+bIMNNsiBBx6YAw88MO+//36efPLJ/OpXv8pJJ52U/fffP8cee2xrzQkAAADAKmyFIlOzG3bqlH79+qVfv34ZMWJEFi1atDLnAgAAAKAdafGcTEny4IMP5tprr80f/vCHvP/++znttNOy/fbbZ8iQIZk+fXqSpHPnzq06KAAAAACrrhYj0xVXXJHzzjsv06ZNy+mnn55TTz01VVVV+c///M9suummufDCC9tiTgAAAABWYS1+XO62227LL37xi3Tv3j2vvPJK9t577zz++ONZa621UltbmwEDBrTFnAAAAACswlo8kumtt95K9+7dkyS9evXKGmuskbXWWitJsuaaa2bhwoWtOyEAAAAAq7zlOifTh3Xs2LE15gAAAACgHWvx43Lvvvtudt111/Llt956q3y5VCrlvffea63ZAAAAAGgnWoxMEyZMaIs5AAAAAGjHWoxMX/rSl9piDgAAAADasRYj0/jx41u8kxNOOGGlDAMAAABA+7Rckelzn/tc/v3f/z2lUmmJ/VVVVa0yGAAAAADtR4uR6eyzz87dd9+dZ599NnV1damrq8tGG23UFrMBAAAA0E50aOkKw4YNyx133JHLLrss//znPzNkyJAceeSRufvuu7Nw4cK2mBEAAACAVVyLkekDX/jCF3LGGWfk/vvvT58+fXL22WfnySefbM3ZAAAAAGgnWvy43Adeeuml3Hnnnbnnnnuy6aab5sILL8z222/fmrMBAAAA0E60GJluuOGG3HXXXXnvvfdSV1eXG2+8Md26dWuL2QAAAABoJ1qMTBdeeGE+97nPZeutt86LL76Y//zP/1ziOuPGjWuV4QAAAABoH1qMTN/5zndSVVXVFrMAAAAA0E61GJlOPPHEtpgDAAAAgHZsmb9d7rnnnluuO1ne6wEAAADw6bTMI5nOPffcrLXWWqmrq8sXv/jFbLTRRuV9s2bNyuOPP5677ror8+fPzy9+8YtWHxYAAACAVdMyI9NNN92Uhx56KDfffHPOOeecdOjQIWuuuWbmz5+fJNlpp51y2GGHZZdddmmTYQEAAABYNbV4Tqbddtstu+22WxYtWpS///3vaWhoyDrrrJPPfvaz6dy5c1vMCAAAAMAqrsXI9IHOnTvnC1/4QrNt//znPzNp0qQMHTp0pQ8GAAAAQPuxzBN/L01jY2MefPDBnHjiifnqV7+am2++uTXmAgAAAKAdWe4jmZ599tnceeedueeee/Lee+9l4cKFufzyy9O/f//WnA8AAACAdqDFI5l+/OMfZ7/99suQIUPy6quv5pxzzskjjzySmpqabLvttm0xIwAAAACruBaPZLr00ktTU1OTsWPHZu+9905VVVVbzAUAAABAO9LikUwTJkzIbrvtlpEjR2bnnXfOxRdfnKlTp7bFbAAAAAC0Ey1Gpn79+uWiiy7KI488ktNOOy3PP/98Dj744MyePTs333xz5s6d2xZzAgAAALAKW+4Tf6+++urZf//9s//++2fGjBm5++67c9ddd+Xaa6/NM88805ozAgAAALCKW+7I9GHdunXL8OHDM3z4cIEJAAAAgJYj0zvvvJMkWWONNZIkpVIpt956a1544YX07ds3++67b+EhHnrooVx22WUplUoplUo54YQTsscee+Tll1/OWWedlXnz5pVPPt6rV6/CjwcAAADAytXiOZlOOeWU3HfffeXLY8eOzf/z//w/mTVrVi644IL89Kc/LTRAqVTKmWeemXHjxuXuu+/OuHHjMmLEiDQ1NWX06NE59NBDM3ny5Bx66KEZNWpUoccCAAAAoHW0GJmeffbZ9O/fP0mycOHC3HLLLbnsssty+eWX55prrsktt9xSfIgOHfLWW28lSd56661suOGGmTt3bqZNm5aBAwcmSQYOHJhp06Zlzpw5hR8PAAAAgJWrxY/Lvfvuu6murk6STJ06NZ06dcqOO+6YJNlmm23yxhtvFBqgqqoqP/rRj/Ltb387a6yxRubPn59rr702M2bMyEYbbZSOHTsmSTp27JgNN9wwM2bMSNeuXZvdR0NDQxoaGpptq6+vLzQXAAAAAMuvxci04YYb5rnnnssWW2yRRx55JDvssEN5X0NDQ7p06VJogPfffz/XXHNNrrrqquywww558skn893vfjfjxo1b7vuYMGFCxo8fX2gOAAAAAD65FiPTUUcdlaOPPjp9+/bNb3/721xxxRXlfb/97W+z+eabFxrgL3/5S2bNmlWOVzvssENWX331rLbaapk5c2YaGxvTsWPHNDY2ZtasWenWrdsS9zFs2LAMHjy42bb6+voMHTq00GwAAAAALJ8WI9NBBx2Unj17ZurUqfmP//iP1NbWlvetttpqOeGEEwoNsPHGG6e+vj5/+9vf8vnPfz4vvfRSZs+enZ49e6ZPnz6ZNGlS6urqMmnSpPTp02eJj8olSXV1dfkjfQAAAAC0vRYjU5J86Utfype+9KUlttfW1mbSpEnNwtOK2mCDDTJmzJicfPLJqaqqSpL84Ac/SE1NTcaMGZOzzjorV111VaqrqzN27NhP/DgAAAAAtJ7likwf1tjYmF/96le566678vDDD6dnz56FP5Y2aNCgDBo0aIntvXv3zq233lrovgEAAABofcsdmZ599tnceeedueeee/Lee+9l4cKFufzyy9O/f//WnA8AAACAdqBDS1f48Y9/nP322y9DhgzJq6++mnPOOSePPPJIampqsu2227bFjAAAAACs4lo8kunSSy9NTU1Nxo4dm7333rt83iQAAAAA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xdqjkqw/KsXao5KsPyrF2vvkevbsudTtq/SJv9dbb7306dMnkyZNSpJMmjQpffr0aXY+JgAAAAAqb5U+kilJxowZk7POOitXXXVVqqurM3bs2EqPBAAAAMBHrPKRqXfv3rn11lsrPQYAAAAAy7BKf1yOyqiurs4JJ5yw1PNcQWuz/qgUa49Ksv6oFGuPSrL+qBRrr/VUlUqlUqWHAAAAAKB9cyQTAAAAAIWJTAAAAAAUJjIBAAAAUJjI1I7985//zDbbbJMLLrigvO2OO+5IbW1t6urqyn8uvfTS8vVPPfXUDBw4MPvtt1/q6ury+9//PklyxRVXZKeddspJJ51Uvq/TTjstX/3qV7P55ptn/vz55e0vv/xyDj/88Oy1114ZOHBgzj777Lz33nvl/VdffXX22WefDBo0KN/85jfz17/+NUny6quvpq6uLltvvXVeeOGFVv3a/KtbmWvj4/zf//1f9t9//9TV1WWvvfbKaaedVt7Xv3//Jb7HBxxwQB577LFm27773e9mxx13zKJFi5pt33zzzctz1NXV5fnnn0+SLFy4MEcffXT69euXfv36NbvNH//4xwwZMiT77LNP9tlnn4wdOzYfnHLug+c+ZMiQJMncuXNz7LHHZs8998x+++2XE044IXPmzGnx8ZPF6/iD2+6zzz7l335pfbedSr32Jcntt9+e/fbbL/vuu2+GDx+eefPmJVn266K18enRmmuvpTW05ZZbNnuMuXPnlmf4y1/+kqFDh5Zf/x5++OEki18X6+rqlrqWWbna8/tuU1NTDjnkkAwaNCiDBg3K0UcfnVdffbW8/7bbbivPeMABB+SJJ55YYrazzz672Tp77LHHsu2226auri4NDQ1pamrKiSeemD333DODBg3KkUcemX/84x/N5t9rr73KX6ff/OY35X3z5s3Lqaeemj333DP77rtvxo8fn2Tx3wnq6urSt2/fPPTQQ8v82rFyfPB9GjRoUL7+9a/n+OOPzx//+MckK/d9+OO+5x/20TXnvbb9aI11tCLOOuus/PznP2/xeocffngGDBiQa6+9Nkkyc+bMHH744dlhhx1ywAEHNLtuY2NjLrzwwgwcODB77rlns59DHnjggRxwwAEZOHBg9t133/z0pz8t3+7ee+/NPvvss8TPNe1eiXbr5z//eemwww4r7bjjjqUFCxaUSqVS6fbbby+deOKJS73+mDFjShdffHGpqampVCqVSnPmzCm99tprpVKpVLr88stLF198cbPr/+53vyu9+eabpc0226z09ttvl7dPnz699Oyzz5ZKpVKpsbGxdPLJJ5fGjx9fKpVKpWnTppV23XXX0vz580ulUqk0YcKE0jHHHNPsfnfbbbfS888/X/Tpswwrc20szcyZM0v9+vUrvf7666VSqVRqamoqr4lSaenf48GDB5ceffTR8uW5c+eWvvjFL5YOPvjg0r333tvsuh9dcx9YtGhR6ZFHHilNmzat9KUvfanZvueff7708ssvl0qlUmnBggWlIUOGlO68886lPve5c+c2m+Xiiy8unX322S0+flNTU6murq50//33ly+/+eabza5jfbe+Sr32vfjii6WvfvWrpdmzZ5dKpVLpyiuvLH3/+98vlUrLfl38gLXR/rXm2lvWGpo+ffoSr3kfmD9/fql///6lp556qlQqLX6dnDNnTrPrfNxrGitPe3/fbWhoKP/3f//3f5e+853vlOfq27dv6Y033iiVSqXSAw88UNp7772b3fbBBx8snX322c3W2aOPPloaPHhw+TqNjY2lBx54oNTY2FgqlUqlG264oXTEEUcsc/4PfOtb3ypdf/315cuzZs1qtv+www4rTZkyZam3ZeX66Pdp8uTJpR122KH09NNPr9T34Za+50tbcx83I6ue1lhHK2LEiBGlG264ocXrffS1paGhofT444+XHnrooWavb6VSqXTzzTeXjjrqqNLChQtLixYtKh199NGlSZMmlUqlUunpp58u1dfXl+9j9913Lz3++OPl2y7rPb69ciRTO3b77bfn29/+djbffPM8+OCDLV6/vr4+G220UaqqqpIk6667bjbZZJOPvf5OO+2U9dZbb4ntPXr0yJZbbpkk6dChQ7bZZpu8/vrrSZKqqqosWrSo/K+vb731VjbeeOMVfm4U09pr480330ynTp1SU1OTZPH3/YM1sbwmTpyYXXbZJYceemhuv/325bpNp06d8uUvfzlrr732Evs222yz9OrVK0nSpUuXbLnlluV1+VE1NTXN/sVgu+22+9jrftjvfve7rLnmmtl9992TLH7eS/t/hNZVqde+F154IX369EnXrl2TJLvssksmTpyYZNmvi3x6tOba+6RraNKkSdlhhx2y3XbbJVn8Ornuuusu5zNiZWnv77sffl99++2306HD4h8RSqVSSqVS+WiRj/69bu7cuRk/fnzOPvvsZT52hw4dMmDAgPL9Lu/77iuvvJIXXnghw4YNK2/bYIMNWrwdbWOPPfbIkCFD8pOf/GSZ11uR9d7S93x51xztx8pYRwsXLszYsWNz4IEHZtCgQTnjjDPKr1szZ87MsGHDss8+++TYY49tdiTw22+/nXPOOScHHnhg9ttvv1xwwQVpbGxc6uOvvfbaqa2tzeqrr77Evueeey477bRTOnfunE6dOuUrX/lK+e+I2267bTbaaKPyffTu3TuvvfbaCn6V2pdOlR6AT+a5557LvHnzsuOOO+aNN97I7bffnr333jvJ4h+E6+rqytc97LDDctBBB+WII47ISSedlEmTJqVv377p379/dtppp0JzvPfee7n99ttz6qmnJkm22GKLHHnkkenfv3/WXnvtVFdXL9fhiKw8bbE2tthii2yzzTbZdddd069fv2y//fapq6tr9oPNSSedlNVWW618+ZVXXml2H7fffntGjBiR7bbbLhdeeGFmzpxZfgFOFh+i2tjYmJ133jknnnhiunTpstxfg9mzZ2fy5Mnlw1uXpampKTfddFP69+/fbPvSHv/FF19MTU1NTjrppPzjH//IZz/72Zx99tnp1q3bcs9GMZV87dtiiy3y5z//OdOnT0+PHj0yadKkvPPOO5k3b175B79kyddFPh3acu0tbQ3Nnz+/fHj+Pvvsk6OPPjpVVVV58cUX06lTpxx77LGZNWtWttpqq4wYMSLrrLPOSv4K8HE+Le+7xx57bKZNm5Z11123/MNe165dc95552Xw4MGprq5OU1NTbrjhhvJtzjvvvJx00klL/cefZbnxxhuXeN89/fTTUyqVssMOO+TUU09NdXV1XnzxxWy00UY555xz8pe//CXrr79+zjzzzPzbv/3bCj0erWfbbbfNlClTsuuuu66U9d7S9/yTrjlWbUXX0Y9//OOsvfbaue2225Ikl1xySa699tqccsopueCCC/LFL34xJ5xwQqZPn55Bgwbla1/7WpLkoosuyhe/+MVceOGFaWpqyumnn57bb789Bx988ArNv9VWW+WOO+7IoYcemmTxR+QaGhqWuN5LL72Up59+Oueee+4n+jq1FyJTO3Xbbbelrq4uVVVV2WOPPXLBBRdk5syZSZIvf/nLufzyy5e4zU477ZSHHnoojz32WJ588sl897vfzdFHH53jjjvuE83w/vvv55RTTsmOO+6YAQMGJElee+21PPjgg7nvvvuy4YYb5sc//nHOOuusXHPNNZ/8ybJC2mJtdOjQIVdddVVeeOGFPP7443nggQfyk5/8JBMnTiz/sH355Zdns802K9/mw59dnjZtWhoaGrLjjjuW57zrrrvyrW99K0nyq1/9Kt26dcvbb7+dM844I1deeWVOOeWU5Xr+b7/9do4//vgcddRRy/WvvOeff37WWGONHHbYYeVtH/f4TU1NefTRR3PLLbekd+/euf766zNixIj87Gc/W67ZKK6Sr32f+9znMnLkyJxyyimpqqoqv+516vT/v5Uu7XWRT4e2WntLW0MbbrhhHn744ay33nqZPXt2jj/++Kyzzjo56KCDyq9LN998c9Zff/1cdNFFufjii3PRRRe1zheCJXwa3neT5LrrrktTU1Ouueaa/Nd//VfGjBmTt99+OzfeeGNuu+22fP7zn88999yTE044If/7v/+b//u//0vnzp2z6667rtDX67rrrstLL72UCRMmlLfdeOON6datWxYuXJgLL7ww5513Xi699NI0NTXlmWeeyWmnnZba2trcd999Of744/PAAw+s0GPSekr/33lnkpWz3pf1Pb/nnns+0Zpj1Vd0HU2ZMiVvv/12Jk+enGTxkU1bbLFFksXniBs5cmSSZNNNN20WOKdMmZI//elPuf7665Ms/keeD8f35XXAAQdk+vTp+eY3v5m11lor22yzTR599NFm15k1a1a+/e1vZ/To0Z/oMdoTkakdWrhwYSZNmpQuXbrk7rvvTpIsWrQod9xxR4sLdq211sqAAQMyYMCAbL311vmv//qvTxSZGhsbc/rpp2edddYp/0+bLD552WabbZYNN9wwSbL//vsv9WR9tI62XhubbbZZNttss/IJZ//whz9kjz32aHHO22+/PQ0NDeUfoBYuXJg111yz/JfdD44MWmuttXLQQQeVX/hb8u6772b48OH5yle+kqOOOqrF648dOzZ///vfc/XVV5cP4V/W43fr1i1bbbVVevfunSQZNGjQUt8EaR2rwmvfvvvum3333TdJ8qc//Sm/+MUvstZaayX5+NdF2r+2Wnsft4a6dOlS/gjneuutl/322y9//OMfc9BBB6Vbt27p169f+X13v/32y/e+972V8bRZDp+W990PdOjQIQceeGD22GOPjBkzJr/97W+z9tpr5/Of/3ySxUfRnX322Zk7d27+8Ic/5NFHH212RNLAgQNz3XXXfewcN9xwQyZNmpQJEyY0+8jJB++7Xbp0yaGHHprjjz++vL1bt26pra1NsvhjNWeccUbmzJlT/ugylfXnP/95uY4sW971vqzv+bLW3Be+8IWV96Roc0XXUalUyujRo1f4SPVSqZSrrroqm2666ScdPcni185TTjml/I/i1113XfnnhWTxpyyOPPLIHHPMMeUjXT/NnJOpHXrwwQfzuc99Lr/+9a8zZcqUTJkyJT/96U9z5513LvN2jzzySN5+++0ki/+HmjZtWnr06LHCj9/U1JSzzjorHTt2zIUXXlj+XGyy+JwSTz75ZN55550kycMPP+yQ5jbUVmtj5syZeeqpp8qX6+vrM2fOnOVaTx/8hfz2228vz/jb3/42SfLEE0/kn//8Z/mcXu+//34mT56cPn36tHi/CxYsyPDhw7Ptttvm5JNPbvH6P/zhDzN16tRceeWVzT6Kt6zH33nnnVNfX59Zs2YlSX7zm99k8803b/GxWDkq/dqXJG+88UaSxevt8ssvL8fMZb0u0v61xdpb1hqaPXt2+beBvfvuu5kyZUr5X2j33nvv/OlPfyo/zq9//WuvS23o0/C+O2fOnGa/YfXee+8tr6EePXpk2rRpmT17dpLk0UcfzVprrZV11103Y8aMafa8k8XnCPu4H/Zvvvnm3HLLLbn++uubfcT4nXfeyVtvvVX+Wtxzzz3l992tt946a6yxRvk3FT/++ONZZ511nHdsFfHAAw/kpptuavEf9lZkvS/re76ia472YWWso/79++e///u/y3+Hf/vtt/PSSy8lSXbcccfyeeimT5/e7DfS9e/fP9dee235PExz5szJ9OnTV/g5LFiwoPw69vrrr+emm27KkUcemWTxecSOPPLIDB06NAcddNAK33d75EimduiDX6H9YX379k1TU1Nef/31JT7HuvXWW+fCCy/M888/n4svvrh8OGLPnj0zatSoj32cE044IX/605+SJHvttVc222yz/OQnP8mvf/3r/O///m8222yz8qHY22+/fUaPHp099tgjzzzzTA444IB06dIl1dXVDtlvQ221Nt5///1cccUVee211/KZz3wmTU1N+e53v7tcH0974IEH8tnPfjY9e/Zstn2//fYrfwZ61KhRqaqqyvvvv5++ffs2i0bf+MY3MnPmzDQ0NGTnnXfO1772tVx44YW57bbb8oc//CHz5s0r/+V5r732Kv9r6If99a9/zTXXXJNevXplyJAhSRb/RfrKK6/M3/72t499/DXWWCMjR47Msccem1KplJqamlx88cUtPmdWjkq/9iWLf2Xy66+/nkWLFmWfffbJEUcckSTLfF2k/WuLtbesNfTkk0/m8ssvT4cOHfL+++9n1113LX/Ed5NNNsmxxx6bIUOGpKqqKj169Mj555/fGl8GluLT8L77H//xHzn77LPLIbN79+655JJLyvMec8wxOeyww9K5c+d06dIll1122QqH9LfffjtjxozJJptsUv7Bq0uXLrn11lsze/bsnHjiiWlsbExTU1N69+5dfu2sqqrKD37wg5x99tlZuHBhVl999YwfP17Ir6CTTjopXbp0ybvvvpvevXvn2muvzbbbbpuXXnpppax33/N/DSt7HR133HEZP358DjzwwFRVVaWqqionnHBCevfunXPOOSdnnnlmJk2alB49ejT75T/f+973cskll5Q/8ty5c+d873vfW+qRTY2Njdltt92ycOHCvP3229l5551z0EEH5cQTT8xbb72Vww8/vPzJiNNPPz1bbbVVkuTaa6/NK6+8kv/5n//J//zP/yRJjjjiiHzjG99onS/uKqCq9OEPQPIv64orrsg777yTESNGtPpj9e/fP1dffXWz8wZAa7rjjjvyq1/9qk0+2mZ9ty9e+6iUtlx7m2++ef74xz9mzTXXbPXHgmTxOVDGjh2bO+64o9Uf6/DDD89RRx2V3XbbrdUfi5VvZb8Weq9lZWmr15ZXX3013/jGN/LYY4+16uO0JR+XI8niIzTuv//+nHTSSa32GK+++mrq6uqyaNGiZifKhdb2mc98JlOnTi0ftdQarO/2yWsfldIWa++Pf/xj6urqsv766zc77xy0ts6dO2f27Nmpq6tb6m9YWhkWLlyYurq6TJ8+vdlv1aN9WVmvhd5rWdnWWWedjBs3brl+W/Unde+99+b444/P+uuv32qPUQmOZAI+1vjx43P//fcvsf2nP/1p+SS0AMDK4X0XgPZOZAIAAACgMMdOAwAAAFCYyAQAAABAYSITAAAAAIWJTAAAn8DEiRNzwAEHpG/fvvnqV7+aY445Jk888USuuOKKnH766eXrbb755tluu+3St2/ffO1rX8tFF12UxsbGJIt/3fbvfve7Zvd7xx135Jvf/Gb5cv/+/bPNNtukb9++5T/nnXde+bp9+vRJ3759s/3222fQoEF56KGH2uDZAwAsye93BABYQddff32uvfbanHvuufnqV7+azp075ze/+U0efPDBrLHGGktc/+67707Pnj3z0ksv5YgjjkivXr2ahaSWXH311fnyl7+81H3bbbddbrrppjQ1NeWWW27JqaeemocffjjV1dWf+PkBAHwSjmQCAFgBb731Vi6//PKMGjUqe+yxR9ZYY4107tw5/fv3z4gRI5Z52969e2eHHXbIX//615U+V4cOHVJXV5d33nknr7zyykq/fwCAlohMAAAr4KmnnsqCBQvy9a9/fYVv++KLL+bJJ59Mnz59VvpcjY2NueOOO9K5c+d07959pd8/AEBLfFwOAGAFzJs3L+uuu246dVr+v0YNHjw4HTt2zDrrrJMDDzww3/jGN8r7vvOd76Rjx47ly4sWLcqWW27Z7PYfvc6ZZ56Zgw8+OEnyzDPPpLa2Nu+++246duyYcePGZb311vukTw8A4BMTmQAAVkBNTU3mzp2b999/f7lD05133pmePXsudd+VV17Z7HxLd9xxR2699dZlXufDtt1229x0002ZP39+zjnnnDz55JPZZ599lvPZAACsPD4uBwCwAvr27ZsuXbrkgQceqPQozay55poZM2ZM7r777kybNq3S4wAA/4JEJgCAFbD22mvnpJNOynnnnZcHHngg7777bhYtWpSHH34448aNq+hsNTU1Oeigg3LllVdWdA4A4F+Tj8sBAKygo446Kuuvv36uuuqqnH766VlzzTWz1VZbZfjw4XnkkUdW+uMNHz682TmZvvzlL39sSBo2bFh23333PPfcc9liiy1W+iwAAB+nqlQqlSo9BAAAAADtm4/LAQAAAFCYyAQAAABAYSITAAAAAIWJTAAAAAAUJjIBAAAAUJjIBAAAAEBhIhMAAAAAhYlMAAAAABQmMgEAAABQ2P8LE9pYTMeUdMkAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "sns.barplot(x=\"CIPHER\",y=\"TASA(MB/SEC)\",data=info_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "civilian-cologne",
   "metadata": {},
   "source": [
    "# Decrypt Mode"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 487,
   "id": "ready-eclipse",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CIPHER</th>\n",
       "      <th>TIME(SEC)</th>\n",
       "      <th>TIME(MIN)</th>\n",
       "      <th>SIZE(BITS)</th>\n",
       "      <th>TASA(BITS/SEC)</th>\n",
       "      <th>SIZE(BYTES)</th>\n",
       "      <th>TASA(BYTES/SEC)</th>\n",
       "      <th>SIZE(MB)</th>\n",
       "      <th>TASA(MB/SEC)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>AES_SHA384[256]</td>\n",
       "      <td>703.895</td>\n",
       "      <td>11.731583</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>2.935839e+07</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>3.669798e+06</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>29.358385</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>AES[256]</td>\n",
       "      <td>709.823</td>\n",
       "      <td>11.830383</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>2.911320e+07</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>3.639150e+06</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>29.113202</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>AES[128]</td>\n",
       "      <td>711.830</td>\n",
       "      <td>11.863833</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>2.903112e+07</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>3.628890e+06</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>29.031118</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>AES[192]</td>\n",
       "      <td>741.676</td>\n",
       "      <td>12.361267</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>2.786287e+07</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>3.482859e+06</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>27.862868</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>AES_SHA512[256]</td>\n",
       "      <td>742.977</td>\n",
       "      <td>12.382950</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>2.781408e+07</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>3.476760e+06</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>27.814079</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>DES[64]</td>\n",
       "      <td>941.495</td>\n",
       "      <td>15.691583</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>2.194937e+07</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>2.743671e+06</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>21.949368</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>DESede[192]</td>\n",
       "      <td>1356.863</td>\n",
       "      <td>22.614383</td>\n",
       "      <td>20665220656</td>\n",
       "      <td>1.523015e+07</td>\n",
       "      <td>2583152582</td>\n",
       "      <td>1.903768e+06</td>\n",
       "      <td>20665.220656</td>\n",
       "      <td>15.230145</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            CIPHER  TIME(SEC)  TIME(MIN)   SIZE(BITS)  TASA(BITS/SEC)  \\\n",
       "5  AES_SHA384[256]    703.895  11.731583  20665220656    2.935839e+07   \n",
       "4         AES[256]    709.823  11.830383  20665220656    2.911320e+07   \n",
       "2         AES[128]    711.830  11.863833  20665220656    2.903112e+07   \n",
       "3         AES[192]    741.676  12.361267  20665220656    2.786287e+07   \n",
       "6  AES_SHA512[256]    742.977  12.382950  20665220656    2.781408e+07   \n",
       "0          DES[64]    941.495  15.691583  20665220656    2.194937e+07   \n",
       "1      DESede[192]   1356.863  22.614383  20665220656    1.523015e+07   \n",
       "\n",
       "   SIZE(BYTES)  TASA(BYTES/SEC)      SIZE(MB)  TASA(MB/SEC)  \n",
       "5   2583152582     3.669798e+06  20665.220656     29.358385  \n",
       "4   2583152582     3.639150e+06  20665.220656     29.113202  \n",
       "2   2583152582     3.628890e+06  20665.220656     29.031118  \n",
       "3   2583152582     3.482859e+06  20665.220656     27.862868  \n",
       "6   2583152582     3.476760e+06  20665.220656     27.814079  \n",
       "0   2583152582     2.743671e+06  20665.220656     21.949368  \n",
       "1   2583152582     1.903768e+06  20665.220656     15.230145  "
      ]
     },
     "execution_count": 487,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "_info_df = pd.DataFrame(_infos,columns=[\"CIPHER\",\"TIME(SEC)\",\"TIME(MIN)\",\"SIZE(BITS)\",\"TASA(BITS/SEC)\",\"SIZE(BYTES)\",\"TASA(BYTES/SEC)\",\"SIZE(MB)\",\"TASA(MB/SEC)\"])\n",
    "_info_df = _info_df.sort_values(\"TIME(SEC)\",ascending=True)\n",
    "_info_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 488,
   "id": "varied-intake",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='CIPHER', ylabel='TIME(SEC)'>"
      ]
     },
     "execution_count": 488,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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9OjMnDkzo0ePzrhx46q3624MAAAAgHVHzQNUktTX1+f5559Pkjz//PPZdttt89xzz2X27NkZPnx4kmT48OGZPXt2Fi5cmAULFqx27JXa29szd+7cLn/a2tp6bucAAAAA3uR613oCdXV1+cIXvpCPfexj2XTTTbNkyZJcf/31mTdvXrbbbrv06tUrSdKrV69su+22mTdvXiqVymrH+vXr1+X+p0yZksmTJ/f4fgEAAACwUs0D1IoVK3LdddflmmuuyT777JMHH3wwH//4xzNp0qS1cv9jx47NyJEju2xra2vLmDFj1sr9AwAAANC9mgeo3/72t3n66aezzz77JEn22WefbLLJJtloo40yf/78dHR0pFevXuno6MjTTz+dpqamVCqV1Y69UkNDQxoaGnp6twAAAAD4XzU/B9T222+ftra2/PGPf0ySPP7441mwYEF22mmnDB48ODNmzEiSzJgxI4MHD06/fv2y1VZbrXYMAAAAgHVLzY+A2mabbTJhwoScccYZqaurS5J87nOfS2NjYyZMmJDzzjsv11xzTRoaGjJx4sTq7bobAwAAAGDd8ZoC1PLly/PEE0+kvb09DQ0Nedvb3pY+ffqstUmMGDEiI0aMeNX2gQMH5lvf+tYqb9PdGAAAAADrjm4D1I9+9KNMnTo1//3f/53evXtns802y5IlS7JixYrst99+GTVqVN73vvf11FwBAAAAWA+tNkCNGjUqW2yxRYYPH57PfOYz2W677apj8+fPz/3335+pU6fmuuuuy9SpU3tksgAAAACsf1YboD7zmc9k0KBBqxzbbrvtMnz48AwfPjyPPvposckBAAAAsP5b7bfgrS4+vdHrAQAAAPDmtNoAlSTTp0/PmWeeucqxs846K7fddluRSQEAAACw4eg2QE2dOjUnnnjiKsdOOumkfOMb3ygyKQAAAAA2HN0GqD/96U/ZbbfdVjm266675sknnywxJwAAAAA2IN0GqM7OzixatGiVY4sWLUpnZ2eJOQEAAACwAek2QA0ZMiTTpk1b5dj06dOz9957l5gTAAAAABuQ3t0NnnrqqRk7dmzmzZuXgw46KNtss02eeeaZ3HnnnZk+fXqmTJnSU/MEAAAAYD3VbYDac88989WvfjWXX355vvGNb6SzszP19fXZe++985WvfCX/8A//0FPzBAAAAGA91W2ASlZ+DO8b3/hGXnrppfzlL3/JFltskY033rgn5gYAAADABqDbc0B9//vfr/73xhtvnCVLlnSJT1/72teKTQwAAACADUO3AerCCy/scnnUqFFdLl911VVrf0YAAAAAbFC6DVCVSuV1XQYAAACAV+o2QNXV1b2uywAAAADwSt0GqGTlUU6dnZ3p6OhY5WUAgNI6li2v9RSokVr/3a9Y7jXvm5m/f4C1p9tvwXvhhRey2267VS9XKpXq5Uql4ggoAKBH9OrbJ9875thaT4MaOOQ/bqjp4/fu0yufu/CWms6B2rngksNrPQWADUa3Aeruu+/uqXkAAAAAsIHqNkD1799/ldv/8pe/ZIsttigyIQAAAAA2LN2eA+rWW2/NT3/60+rlX//613nve9+b/fbbL8OGDcsf//jH4hMEAAAAYP3WbYD6yle+km222aZ6edy4cTnggAPyne98JwcccEAmTZpUfIIAAAAArN+6/QheW1tbdtlllyTJvHnz8thjj+WGG25IY2NjPvGJT+Sggw7qkUkCAAAAsP7q9gioXr16ZfnylV99+/DDD+ftb397GhsbkySbbLJJXnrppeITBAAAAGD91m2A+sd//Mf8v//3//K73/0uN954Y973vvdVx/74xz92+XgeAAAAAKxKtwHqwgsvzOzZs/PRj340m2yySU488cTq2G233ZZ3v/vdxScIAAAAwPqt23NAbbfddvmP//iPVY6dffbZRSYEAAAAwIZltUdAPfvss6/pDl7r9QAAAAB4c1rtEVBjx47NO9/5zrS2tmavvfZKff3/tarOzs786le/yq233poHHnggM2bM6JHJAlBby1YsT9/efWo9DWrA3z0AAH+L1Qaob3/72/nmN7+ZT3/605k7d2523HHHbLbZZlmyZEnmzp2bt771rTnyyCNzwQUX9OR84U2vc8Xy1HsT+KZV67//vr375F9uOKNmj0/tfO3YK2s9BQAA1mOrDVB9+/bNUUcdlaOOOirz5s3LY489lvb29jQ0NGTXXXfNdttt15PzBP5Xfe8+eXDSCbWeBjWyzyf/vdZTAAAAeN26PQn5y5qamtLU1FR6LuuVZcs70rdPr1pPgxrwdw8AAACvT7cB6pRTTsmXvvSl6uWrrroqp59+evXyhz/84UybNq3c7NZhffv0yuhP3lTraVAD35g0ptZTAAAAgPXKar8FL0nuu+++Lpe//vWv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sGH54Ac/mMmTJydJli1bltbW1gwZMiQ//OEPu/3Z8bdZm+v7i1/8Yvbff/+cfvrpSZInnngiRx99dA4++OAMHz48559/fl566aUkydy5c7Pbbrt1eYznnnuuOoff/va3GTNmTA455JAccsgh+fGPf5wkeeihh9La2tplTbL+Krn+kuQTn/hE3vWud71qvXS3NpPk2muvzSGHHJIRI0bkox/9aH7/+98nWbluW1tbs8cee7zqeZn1S63WXpJMmzYthx56aD74wQ/m5JNPzqJFi5Ks+TnT2usZ6/PrviQZNGhQdR6tra159NFHk6x8bXX88cdn3333zb777tvlNg899FBGjRpV/Z07ceLEvHza2Zf3fdSoUUmS5557LieeeGKGDRuWQw89NKeeemoWLly4xsdPVq7jl297yCGHVL8d2/rueS+/Ph8xYkT++Z//OaecckoeeuihJGv3uXB1r/X/2ivfa1gP648S6+j1OO+88/L1r399jdc7+uijM3To0Fx//fVJkvnz5+foo4/OPvvsk8MOO6zLdTs6OnLJJZdk+PDhGTZsWJfnw7vuuiuHHXZYhg8fng9+8IP56le/Wr3dHXfckUMOOeRVz6/rvQo97utf/3rlqKOOquy3336VpUuXViqVSmXatGmV0047bZXXnzBhQuWyyy6rdHZ2ViqVSmXhwoWVP//5z6u9//nz51f23XffylNPPVWpVCqVzs7Oym9+85vq+Pve977Ko48+2uU2I0eOrNx7773Vy88991zlne98Z+UjH/lI5Y477uhy3fb29up/f+1rX6v827/9W3VeQ4YMqTzzzDOVSqVSueuuuyof+MAHutz27rvvrpx//vmVXXbZpbJ48eJKpVKp3HvvvZWRI0dWr9PR0VG56667Kh0dHZVKpVK58cYbK8ccc0y383/Zv/7rv1ZuuOGG6uWnn366y/hRRx1Vueeee1Z5W9aOtbm+r7rqqspll11Wve6cOXOqa7mjo6NyxhlnVCZPnlwd+8d//MdVPsaSJUsqLS0tlYcffrhSqVQqy5cvryxcuLDLdf56TbL+Krn+KpVK5ec//3nl2WeffdV66W5tzp49u3LggQdWlixZUqlUKpUpU6ZUTjjhhC73293zGuuHWq29P/zhD5V3vetdlQULFlQqlUrl6quvrnz605+uVCrdr8uXWXvlre+v+1b3+3H58uWVn/3sZ5XZs2e/6vfvo48+WnniiScqlUqlsnTp0sqoUaMq3/72t1e5788991yXuVx22WWV888/f42P39nZWWltba384Ac/qF5+9tlnu1zH+u45r/xZz5w5s7LPPvtUHnnkkbX6XLim1/qreq+xujmy7imxjl6Pc889t3LjjTeu8XqvfE/Z3t5euf/++ys//OEPu7yvrVQqlalTp1aOO+64yrJlyyrLly+vHH/88ZUZM2ZUKpVK5ZFHHqm0tbVV7+P9739/5f7776/etrv3N+srR0DVwLRp0/Kxj30sgwYNyt13373G67e1tWW77bZLXV1dkmTLLbfMDjvssNrrP/vss+ndu3caGxuTJHV1ddltt91e1xxvv/32vPe9783o0aMzbdq0LmObb7559b8XL16c+vqVy6hSqaRSqVT/teH555/P9ttvX73uc889l8mTJ+f888/v9rHr6+szdOjQ6v3uvffeeeqpp9Y45yeffDKPPfZYxo4dW922zTbbrPF2rF0l1/eAAQOqa7m+vj577rnna1obM2bMyD777JO99947SdK7d+9sueWWr3GPWJ+Ufn7df//9s9VWW71qe3drs66uLsuXL68eefLK50Y2DLVae4899lgGDx6cfv36JUne+9735vbbb0/yxp8zWbvW99d9q9O7d+8ccMABXV4XvmyXXXbJzjvvnCTp27dvdtttt9WuvcbGxi7/wv9aX/f9/Oc/z2abbZb3v//9SVbu96r+H6E2DjrooIwaNSpf+cpXur3e61nva3qt/1rfa7D+WBvraNmyZZk4cWIOP/zwjBgxIuecc071/er8+fMzduzYHHLIITnxxBO7fHpi8eLFufDCC3P44Yfn0EMPzcUXX5yOjo5VPv7mm2+e5ubmbLLJJq8a+93vfpf9998/ffr0Se/evfNP//RP1d/Te+21V7bbbrvqfQwcODB//vOfX+dPaf3Su9YTeLP53e9+l0WLFmW//fbLM888k2nTpuUDH/hAkpW/SFtbW6vXPeqoo3LEEUfkmGOOyemnn54ZM2ZkyJAhaWlpyf7777/ax9h1112z55575sADD8y+++6bd7zjHWltbe3yhvv000/PRhttVL385JNPdrmPadOm5dxzz83ee++dSy65JPPnz6/+z5EkJ554YmbPnp0tt9yy+oTQr1+/XHTRRRk5cmQaGhrS2dmZG2+8sXqbiy66KKeffvoqX6h056abbkpLS0uXbWeffXYqlUr22WefnHXWWWloaMgf/vCHbLfddrnwwgvz29/+NltvvXU++clP5u///u9f1+PxxvXE+n7ZSy+9lGnTpuWss86qbluyZEn1sNdDDjkkxx9/fOrq6vKHP/whvXv3zoknnpinn346u+++e84999xsscUWa/knQC315PrrzivX5q677ppjjz02LS0t2XzzzdPQ0PCaDu9m/VHLtbfrrrvm17/+debMmZMBAwZkxowZeeGFF7Jo0aJqkEhW/ZxJeRvK676jjz46HR0dec973pPTTjstffv2fc0/gwULFmTmzJnVj6p0p7OzMzfffPOrXvet6vH/8Ic/pLGxMaeffnr+53/+J29961tz/vnnp6mp6TXPjbL22muv3HPPPTnwwAPXynpf02v9N/peg3Xb37qO/v3f/z2bb755brnlliTJ5Zdfnuuvvz5nnnlmLr744rzzne/Mqaeemjlz5mTEiBF597vfnSS59NJL8853vjOXXHJJOjs7c/bZZ2fatGn5yEc+8rrmv/vuu2f69OkZPXp0kpUfu2tvb3/V9R5//PE88sgj+cxnPvOGfk7rCwGqh91yyy1pbW1NXV1dDjrooFx88cWZP39+kuSAAw7IVVdd9arb7L///vnhD3+Y++67Lw8++GA+/vGP5/jjj89JJ520yseor6/PNddck8ceeyz3339/7rrrrnzlK1/J7bffXn0xetVVV2WXXXap3uavP6s6e/bstLe3Z7/99qvO89Zbb82//uu/Vq/z5S9/OZ2dnbnuuuvypS99KRMmTMjixYtz00035ZZbbsnb3/72fO9738upp56a73znO/n+97+fPn365MADD3xdP68vf/nLefzxxzNlypTqtptuuilNTU1ZtmxZLrnkklx00UW54oor0tnZmV/+8pf5xCc+kebm5tx555055ZRTctddd72ux+SN64n1nSQrVqzImWeemf322y9Dhw5Nkmy77bb58Y9/nK222ioLFizIKaecki222CJHHHFEOjs7c++992bq1KnZeuutc+mll+ayyy7LpZdeWuYHQU301PrrzqrW5p///OfcfffdufPOO7Ptttvm3//933Peeefluuuue+M7yzqllmvvbW97Wz71qU/lzDPPTF1dXXXd9e79fy/xVrUu6Rkbwuu+H/3oR2lqasrixYtzzjnn5Oqrr86ZZ575mvZ/8eLFOeWUU3Lccce9pqOyPvvZz2bTTTfNUUcdVd22usd/+Xf7N7/5zQwcODA33HBDzj333PzHf/zHa5ob5VX+9zw3ydpZ79291v/e9773ht5rsO77W9fRPffck8WLF2fmzJlJVh4RteuuuyZZeS7iT33qU0mSHXfcsUv8vOeee/KrX/0qN9xwQ5KV/5Dz12H+tTrssMMyZ86cfPSjH81b3vKW7Lnnnrn33nu7XOfpp5/Oxz72sYwfP/4NPcb6RIDqQcuWLcuMGTPSt2/f3HbbbUmS5cuXZ/r06WtcaG95y1sydOjQDB06NHvssUe+9KUvrfFF6i677JJddtmleuLlX/ziFznooIPWOM9p06alvb29+iJ12bJl2WyzzboEqGTlC57DDz88Bx10UCZMmJD/+q//yuabb563v/3tSVYegXL++efnueeeyy9+8Yvce++9Xf5Fa/jw4fnyl7+82nnceOONmTFjRqZMmdLlcMaX/2Wrb9++GT16dE455ZTq9qampjQ3NydZecjmOeeck4ULF1Y/mkA5PbW+Ozo6cvbZZ2eLLbao/sJIVq6Hlw+932qrrXLooYfmoYceyhFHHJGmpqbsu+++2XbbbZMkhx56aC644IK1sdusI3r6+XVVVrc277jjjuyyyy7V9fehD31olSdNZf20Lqy9D37wg/ngBz+YJPnVr36Vb3zjG3nLW96SZPXrkvI2lNd9L7/uestb3pIjjjii+mZsTV588cWcfPLJ+ad/+qccd9xxa7z+xIkT86c//SnXXntt9TQM3T1+U1NTdt999wwcODBJMmLEiFW+MaV2fv3rX7+mTyK81vXe3Wv97t5r/N3f/d3a2yl63N+6jiqVSsaPH/+6jzKuVCq55pprsuOOO77RqSdZ+Z75zDPPrIb7L3/5y9XnrWTlUaLHHntsTjjhhOoRshsy54DqQXfffXfe9ra35Sc/+Unuueee3HPPPfnqV7+ab3/7293e7mc/+1kWL16cZOX/CLNnz86AAQNWe/358+fn4Ycfrl5ua2vLwoULu73Ny15+sTRt2rTqHP/rv/4rSfLAAw9k4cKFXb6Z5I477sigQYOSrDzXxOzZs7NgwYIkyb333pu3vOUt2XLLLTNhwoQu+52sPC/P6n4hTJ06Nd/85jdzww03dPkIwQsvvJDnn3+++rP43ve+l8GDBydJ9thjj2y66abVb5e6//77s8UWWzjXTw/pifXd2dmZ8847L7169coll1xS/Zx3svLJ++Vv7nnxxRdzzz33VP914wMf+EB+9atfVR/nJz/5SXXdsmHoqefX1elubQ4YMCAPPvhgXnjhhSTJj3/8Yx8N3oDUeu0lyTPPPJMkWbp0aa666qrqm/3u1iXlbQiv+/7yl79Uz1+3YsWKzJw5s/q6qztLly7NySefnL322itnnHHGGq//+c9/PrNmzcrVV1/d5eN93T3+e97znrS1teXpp59Okvz0pz/1u30dctddd+Xmm29eY3x8Peu9u9f6r/e9BuuHtbGOWlpa8rWvfa36XLJ48eI8/vjjSZL99tuvet67OXPmdPnmvJaWllx//fXV8z4tXLgwc+bMed37sHTp0ur716eeeio333xzjj322CQrz1t27LHHZsyYMTniiCNe932vjxwB1YNe/prkvzZkyJB0dnbmqaeeetVnWvfYY49ccsklefTRR3PZZZdVDz/caaedMm7cuNU+zooVK/LFL34xf/7zn7Pxxhuns7MzH//4x1/Toc933XVX3vrWt2annXbqsv3QQw/NtGnT8i//8i85//zzq2/0+/fvn8svv7w63xNOOCFHHXVU+vTpk759++bKK6983S94Fy9enAkTJmSHHXao/s/Zt2/ffOtb38qCBQty2mmnpaOjI52dnRk4cGDGjx+fZOXJJz/3uc/l/PPPz7Jly7LJJptk8uTJXnD3kJ5Y3z/5yU/yne98J7vsskv14wPveMc7Mn78+Dz44IO56qqrUl9fnxUrVuTAAw+sHsK/ww475MQTT8yoUaNSV1eXAQMG5LOf/WyJHwM10lPPr6eeemp+9atfJUkOPvjg7LLLLvnKV77S7do86KCD8stf/jKHHXZY+vbtm4aGBh//3IDUeu0lK79y/Kmnnsry5ctzyCGH5JhjjknS/XMm5W0Ir/s+8pGPZNy4camrq8uKFSsyZMiQLkHpwx/+cObPn5/29va85z3vybvf/e5ccsklueWWW/KLX/wiixYtqgatgw8+uHrU+l/7/e9/n+uuuy4777xzRo0alWRluL/66qvzxz/+cbWPv+mmm+ZTn/pUTjzxxFQqlTQ2Nuayyy5b4z5Tzumnn56+ffvmxRdfzMCBA3P99ddnr732yuOPP75W1rvX+m8Oa3sdnXTSSZk8eXIOP/zw1NXVpa6uLqeeemoGDhyYCy+8MJ/85CczY8aMDBgwoMsXIlxwwQW5/PLLqx+j7tOnTy644IJVHhHV0dGR973vfVm2bFkWL16c97znPTniiCNy2mmn5fnnn8/RRx9dPbLz7LPPzu67754kuf766/Pkk0/mP//zP/Of//mfSZJjjjkmH/7wh8v8cNcBdZW//lAl1MB9992XiRMnZvr06cUf6+ijj85xxx2X973vfcUfi7/dF7/4xbzwwgs599xziz/WoEGD8tBDD2WzzTYr/lisH3py/bW0tOTaa6/tco4W3rysPTZk06dPz49+9KMe+bic9b1+W9vPhdYDa0tPvaecO3duPvzhD+e+++4r+jg9yUfwqLk+ffpkwYIFaW1tXeU3AqwNy5YtS2tra+bMmdPlW2BYt2266ab5wQ9+kNNPP73YYzz00ENpbW3N1ltv3eWcE9AT62/u3LlpbW3N8uXLu5w0mjc3a48N2cYbb5xZs2ZVj3YqwfreMKyt50LrgbVtiy22yKRJk17Tt3u+UXfccUdOOeWUbL311sUeoxYcAbUemzx5cn7wgx+8avtXv/rV6smYAQBY/3ndB8D6ToACAAAAoCifNwEAAACgKAEKAAAAgKIEKAAAAACKEqAAANay22+/PYcddliGDBmSd73rXTnhhBPywAMP5Itf/GLOPvvs6vUGDRqUvffeO0OGDMm73/3uXHrppeno6Eiy8ivDf/7zn3e53+nTp+ejH/1o9XJLS0v23HPPDBkypPrnoosuql538ODBGTJkSN7xjndkxIgR+eEPf9gDew8A8Gq+hxIAYC264YYbcv311+czn/lM3vWud6VPnz756U9/mrvvvjubbrrpq65/2223Zaeddsrjjz+eY445JjvvvHOXyLQm1157bQ444IBVju299965+eab09nZmW9+85s566yz8uMf/zgNDQ1veP8AAN4IR0ABAKwlzz//fK666qqMGzcuBx10UDbddNP06dMnLS0tOffcc7u97cCBA7PPPvvk97///VqfV319fVpbW/PCCy/kySefXOv3DwCwJgIUAMBa8vDDD2fp0qX553/+59d92z/84Q958MEHM3jw4LU+r46OjkyfPj19+vRJ//791/r9AwCsiY/gAQCsJYsWLcqWW26Z3r1f+0uskSNHplevXtliiy1y+OGH58Mf/nB17N/+7d/Sq1ev6uXly5dnt91263L7V17nk5/8ZD7ykY8kSX75y1+mubk5L774Ynr16pVJkyZlq622eqO7BwDwhglQAABrSWNjY5577rmsWLHiNUeob3/729lpp51WOXb11Vd3Ob/T9OnT861vfavb6/y1vfbaKzfffHOWLFmSCy+8MA8++GAOOeSQ17g3AABrj4/gAQCsJUOGDEnfvn1z11131XoqXWy22WaZMGFCbrvttsyePbvW0wEA3oQEKACAtWTzzTfP6aefnosuuih33XVXXnzxxSxfvjw//vGPM2nSpJrOrbGxMUcccUSuvvrqms4DAHhz8hE8AIC16LjjjsvWW2+da665JmeffXY222yz7L777jn55JPzs5/9bK0/3sknn9zlHFAHHHDAaiPT2LFj8/73vz+/+93vsuuuu671uQAArE5dpVKp1HoSAAAAAGy4fAQPAAAAgKIEKAAAAACKEqAAAAAAKEqAAgAAAKAoAQoAAACAogQoAAAAAIoSoAAAAAAoSoACAAAAoCgBCgAAAICi/j8A5HZ96k7lrwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x=\"CIPHER\",y=\"TIME(SEC)\",data=_info_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 489,
   "id": "separated-entrepreneur",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='CIPHER', ylabel='TASA(MB/SEC)'>"
      ]
     },
     "execution_count": 489,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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oBAhCQAAAIBChCQAAAAAChGSAAAAAChESAIAAACgECEJAAAAgEKEJAAAAAAKEZIAAAAAKERIAgAAAKAQIQkAAACAQoQkAAAAAAoRkgAAAAAoREgCAAAAoJB2LXGQ+fPn55xzzsnLL7+cDh06ZNNNN83o0aOzzjrr5Mknn8zIkSPzwQcfZOONN86ll16addddtyXGAgAAAGAFtMgZSVVVVTnhhBNyzz335M4778wmm2ySyy67LA0NDTn77LMzcuTI3HPPPenTp08uu+yylhgJAAAAgBXUIiGpc+fO6du3b/l2r1698vrrr2fGjBlZbbXV0qdPnyTJwIEDc/fdd7fESAAAAACsoBb5aNvHNTQ05MYbb0y/fv3yxhtv5Ktf/Wr5vnXWWScNDQ15++2307lz50aPq62tTW1tbaNts2bNaomRAQAAAEgFQtIFF1yQjh075qijjsp9991X+HETJ07M+PHjm3EyAAAAAJanRUPS2LFj849//CNXXXVV2rRpky5duuT1118v3z9v3ry0adNmqbORkmTIkCEZMGBAo22zZs3K4MGDm3tsAAAAANKCIennP/95ZsyYkQkTJqRDhw5Jkq222irvv/9+/vznP6dPnz6ZNGlS9t1332U+vrq6OtXV1S01LgAAAACf0CIh6bnnnsvVV1+dzTbbLAMHDkySdO3aNVdccUUuueSSjBo1Kh988EE23njjXHrppS0xEgAAAAArqEVC0re+9a0888wzy7xv2223zZ133tkSYwAAAADwObSp9AAAAAAArBqEJAAAAAAKEZIAAAAAKERIAgAAAKAQIQkAAACAQoQkAAAAAAoRkgAAAAAoREgCAAAAoBAhCQAAAIBChCQAAAAAChGSAAAAAChESAIAAACgECEJAAAAgEKEJAAAAAAKEZIAAAAAKERIAgAAAKAQIQkAAACAQoQkAAAAAAoRkgAAAAAoREgCAAAAoBAhCQAAAIBChCQAAAAAChGSAAAAAChESAIAAACgECEJAAAAgEKEJAAAAAAKEZIAAAAAKERIAgAAAKAQIQkAAACAQoQkAAAAAAoRkgAAAAAoREgCAAAAoBAhCQAAAIBChCQAAAAAChGSAAAAACikXVM7LFmyJNOnT88DDzyQp59+Ou+++27WXHPNbL755tl1112z1157pV27Jr8MAAAAAKu45RagG2+8MVdffXW6deuW7bffPnvssUc6deqUhQsX5oUXXsjNN9+ciy++ON///vdz5JFHttTMAAAAAFTAckPSyy+/nJtvvjnrr7/+Uvd997vfzUknnZQ5c+bkuuuua7YBAQAAAGgdlhuSzj333Ca/wAYbbFBoPwAAAABWbU1ebPu5557LNddcs8z7rrnmmrzwwgsrfSgAAAAAWp8mQ9IVV1yRLl26LPO+jTfeOFdcccVKHwoAAACA1qfJkPTkk0/mu9/97jLv22uvvfLYY4+t9KEAAAAAaH2aDEnvvPNO2rRZ9m5VVVWpra1d6UMBAAAA0Po0GZK6du2aJ554Ypn3PfHEE9l4441X+lAAAAAAtD5NhqTDDjssI0aMyIwZMxptf+qpp/KTn/wkRxxxRLMNBwAAAEDr0a6pHY455pi8/PLLOfzww7PRRhtlgw02yJw5czJ79uwceeSROfroo1tiTgAAAAAqrMmQlCQjRozI0Ucfnf/5n//J22+/nc6dO2ennXbKpptu2tzzAQAAANBKFApJSbLpppsuMxzNmzcv66yzzkodCgAAAIDWp8lrJO2www6Nbg8ZMqTR7b322mvlTgQAAABAq9RkSFq8eHGj23//+98b3S6VSit3IgAAAABapSZDUlVV1ee6HwAAAIAvhiZDEgAAAAAkBS62XVdXl3POOad8+7333mt0u66urnkmAwAAAKBVaTIknXTSSSt0GwAAAIAvpiZD0imnnNIScwAAAADQyjUZkl577bW0bds2G220UZJk0aJFueqqq/Lss8+md+/eOf7449O2bdtmHxQAAACAymryYtvDhw/P3/72t/Lt0aNH57e//W0222yzTJ48Ob/4xS+adUAAAAAAWocmQ9IzzzyT73znO0k+vND2XXfdlf/7f/9vzj333Fx55ZX57W9/2+xDAgAAAFB5TYakxYsXp2PHjkmSv/3tb+nUqVO22mqrJEm3bt0yf/785p0QAAAAgFahyZDUtWvXPPLII0mS6dOnp2/fvuX75s2bl9VXX735pgMAAACg1Sj0W9t++MMfZpNNNsn//u//5vrrry/fN23atPzTP/1Tsw4IAAAAQOvQZEjaa6+9cuutt+bvf/97tthii2yyySbl+77xjW+kV69ezTkfAAAAAK1EkyFp4MCB2X333bPbbrs1ikhJst122zXbYAAAAAC0Lk1eI+m8887L+++/n+HDh2e33XbLiBEjct9992XhwoUtMR8AAAAArUSTZyT16tUrvXr1yo9+9KO8+eabefDBB3PnnXfmJz/5STbffPPstttu2XXXXdOtW7eWmBcAAACACmkyJH3c+uuvn0MPPTSHHnpolixZksceeywPPPBATj311Bx88ME58cQTm2tOAAAAACpshUJSowe2a5e+ffumb9++Offcc7N48eKVORcAAAAArUyT10hKkmnTpmXChAn505/+lCVLluTMM8/Mtttum4EDB+aVV15JkrRv375ZBwUAAACgspoMSZdffnlGjx6dmTNn5qyzzsoZZ5yRqqqq/Ou//ms22WSTjBkzpiXmBAAAAKDCmvxo2y233JLf/OY32XjjjfPSSy9lv/32y6OPPpo11lgjffr0yZ577tkScwIAAABQYU2ekfTuu+9m4403TpJsttlm6dixY9ZYY40kSadOnVJXV9e8EwIAAADQKhS6RtLHtW3btjnmAAAAAKCVa/KjbYsWLcruu+9evv3uu++Wb5dKpbz//vvNNRsAAAAArUiTIWnixIktMQcAAAAArVyTIWmHHXZoiTkAAAAAaOWaDEnjx49v8ouccsopK2UYAAAAAFqvQiHp61//ev7pn/4ppVJpqfurqqqaZTAAAAAAWpcmQ9KwYcNyxx135KmnnkpNTU1qamqy4YYbtsRsAAAAALQibZraYciQIbn11lvzi1/8Iu+8804GDhyYY489NnfccUfq6upaYkYAAAAAWoEmQ9JHvvnNb+bss8/Offfdl549e2bYsGF57LHHCh9o7Nix6devX3r06JFnn322vL1fv37Zd999y2c7/f73v1+x7wAAAABYSsOS+kqPQAU1159/kx9t+8gLL7yQ2267LXfddVc22WSTjBkzJttuu23hA+2555455phjMnjw4KXuGzduXLp37174awEAAADL16Zd2/zlygcqPQYVss0Pdm+Wr9tkSLr++utz++235/33309NTU1uuOGGdOnSZYUP1KdPn880IAAAAACtQ5MhacyYMfn617+erbbaKs8//3z+9V//dal9Lrnkks81xFlnnZVSqZTtttsuZ5xxRqqrq5fap7a2NrW1tY22zZo163MdFwAAAIDimgxJP/zhD1NVVdVsA3x0hlNdXV3GjBmT0aNH57LLLltqv4kTJ2b8+PHNNgcAAAAAy9dkSBo6dGizDvDRx+Q6dOiQQYMG5eSTT17mfkOGDMmAAQMabZs1a9Yyr7kEAAAAwMq33JD09NNPZ/PNN2/yixTd75Pee++91NfXZ80110ypVMpdd92Vnj17LnPf6urqZX7kDQAAAICWsdyQ9NOf/jRrrLFGampqsv3222fDDTcs3zdnzpw8+uijuf3227Nw4cL85je/We6BLrzwwtx777156623cuyxx6Zz58656qqrMnTo0NTX16ehoSHdunXLqFGjVs53BgAAAMBKtdyQdOONN+b+++/PpEmTMnz48LRp0yadOnXKwoULkyQ77bRTjjrqqOy2225NHmjEiBEZMWLEUttvv/32zzY5AAAAAC2qyWsk7bHHHtljjz2yePHi/OMf/0htbW3WWmutfO1rX0v79u1bYkYAAAAAWoEmQ9JH2rdvn29+85uNtr3zzjuZMmWKC14DAAAAfAm0WdEH1NfXZ9q0aRk6dGh23nnnTJo0qTnmAgAAAKCVKXxG0lNPPZXbbrstd911V95///3U1dVl3Lhx6devX3POBwAAAEAr0eQZSddee20OPPDADBw4MK+++mqGDx+ehx56KJ07d84222zTEjMCAAAA0Ao0eUbSZZddls6dO2fs2LHZb7/9UlVV1RJzAQAAANDKNHlG0sSJE7PHHntkxIgR2XXXXXPxxRdnxowZLTEbAAAAAK1IkyGpb9++ueiii/LQQw/lzDPPzDPPPJPDDz88c+fOzaRJkzJ//vyWmBMAAACACit8se3VV189Bx98cA4++OC88cYbueOOO3L77bdnwoQJ+ctf/tKcMwIAAADQChQOSR/XpUuXnHTSSTnppJNEJAAAAIAviSZD0nvvvZck6dixY5KkVCrl5ptvzrPPPpvevXtn//33b94JAQAAAGgVmrxG0umnn5577723fHvs2LH5P//n/2TOnDm58MIL86tf/apZBwQAAACgdWgyJD311FPp169fkqSuri433XRTfvGLX2TcuHG5+uqrc9NNNzX7kAAAAABUXpMhadGiRamurk6SzJgxI+3atcuOO+6YJNl6663z5ptvNu+EAAAAALQKTYakDTbYIE8//XSS5KGHHsp2221Xvq+2tjYdOnRovukAAAAAaDWavNj2cccdl+OPPz69e/fOH/7wh1x++eXl+/7whz+kR48ezTogAAAAAK1DkyHpsMMOy6abbpoZM2bkn//5n9OnT5/yfauttlpOOeWUZh0QAAAAgNahyZCUJDvssEN22GGHpbb36dMnU6ZMaRSXAAAAAPhiavIaSZ9UX1+fadOmZejQodlll10yadKk5pgLAAAAgFam0BlJSfLUU0/ltttuy1133ZX3338/dXV1GTduXPr169ec8wEAAADQSjR5RtK1116bAw88MAMHDsyrr76a4cOH56GHHkrnzp2zzTbbtMSMAAAAALQCTZ6RdNlll6Vz584ZO3Zs9ttvv1RVVbXEXAAAAAC0Mk2ekTRx4sTsscceGTFiRHbddddcfPHFmTFjRkvMBgAAAEAr0mRI6tu3by666KI89NBDOfPMM/PMM8/k8MMPz9y5czNp0qTMnz+/JeYEAAAAoMKaDElTpkxJkqy++uo5+OCDc91112X69Ok57bTTcuedd2b33Xdv7hkBAAAAaAWaDEkjR45cattGG22Uk046KXfffXd+/etfN8tgAAAAALQuTYakUqm03Pv95jYAAACAL4cmf2tbQ0NDHn744eUGpZ122mmlDgUAAABA69NkSKqrq8vw4cM/NSRVVVVl2rRpK30wAAAAAFqXJkPS6quvLhQBAAAA0PQ1kgAAAAAgWQkX2wYAAADgy6HJkPTEE0+0xBwAAAAAtHI+2gYAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFtEhIGjt2bPr165cePXrk2WefLW9/8cUXc8QRR2SfffbJEUcckZdeeqklxgEAAADgM2iRkLTnnnvmhhtuyMYbb9xo+6hRozJo0KDcc889GTRoUEaOHNkS4wAAAADwGbRISOrTp0+6dOnSaNvcuXMzc+bMHHDAAUmSAw44IDNnzsy8efNaYiQAAAAAVlC7Sh34jTfeyIYbbpi2bdsmSdq2bZsNNtggb7zxRtZZZ52l9q+trU1tbW2jbbNmzWqRWQEAAACoYEhaURMnTsz48eMrPQYAAADAl1bFQlKXLl0ye/bs1NfXp23btqmvr8+cOXOW+gjcR4YMGZIBAwY02jZr1qwMHjy4JcYFAAAA+NKrWEhad91107Nnz0yZMiU1NTWZMmVKevbsucyPtSVJdXV1qqurW3hKAAAAAD7SIiHpwgsvzL333pu33norxx57bDp37pzf/va3Of/883PeeeflyiuvTHV1dcaOHdsS4wAAAADwGbRISBoxYkRGjBix1PZu3brl5ptvbokRAAAAAPic2lR6AAAAAABWDUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQSLtKD5Ak/fr1S4cOHbLaaqslSc4666zssssuFZ4KAAAAgI9rFSEpScaNG5fu3btXegwAAAAAPkWrCUlNqa2tTW1tbaNts2bNqtA0AAAAAF8+rSYknXXWWSmVStluu+1yxhlnpLq6utH9EydOzPjx4ys0HQAAAACtIiTdcMMN6dKlS+rq6jJmzJiMHj06l112WaN9hgwZkgEDBjTaNmvWrAwePLglRwUAAAD40moVIalLly5Jkg4dOmTQoEE5+eSTl9qnurp6qbOUAAAAAGg5bSo9wHvvvZd33303SVIqlXLXXXelZ8+eFZ4KAAAAgE+q+BlJc+fOzdChQ1NfX5+GhoZ069Yto0aNqvRYAAAAAHxCxUPSJptskttvv73SYwAAAADQhIp/tA0AAACAVYOQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFCEkAAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABQiJAEAAABQiJAEAAAAQCFCEgAAAACFtIqQ9OKLL+aII47IPvvskyOOOCIvvfRSpUcCAAAA4BNaRUgaNWpUBg0alHvuuSeDBg3KyJEjKz0SAAAAAJ/QrtIDzJ07NzNnzsx1112XJDnggANywQUXZN68eVlnnXXK+9XW1qa2trbRY1977bUkyaxZsz7XDB+89/bnejyrpldffbXSI+TNd9+v9AhUSGtYf++//V6lR6ACWsPam/eB574vo9aw9hYsnF/pEaiQ1rD+3l24qNIjUAGtYe3NqX2r0iNQIZ9n/X3UWerr65e6r6pUKpU+81deCWbMmJFzzz03v/3tb8vb+vfvn0svvTRbbrlledvll1+e8ePHV2JEAAAAgC+dG264IX369Gm0reJnJBU1ZMiQDBgwoNG2urq6vPLKK9lss83Stm3bCk22apo1a1YGDx6cG264IRtttFGlx+FLxvqjUqw9Ksn6o1KsPSrJ+qNSrL3Pp76+Pm+++Wa22mqrpe6reEjq0qVLZs+enfr6+rRt2zb19fWZM2dOunTp0mi/6urqVFdXL/X4b3zjGy016hfSRhttlK5du1Z6DL6krD8qxdqjkqw/KsXao5KsPyrF2vvsNt1002Vur/jFttddd9307NkzU6ZMSZJMmTIlPXv2bHR9JAAAAAAqr+JnJCXJ+eefn/POOy9XXnllqqurM3bs2EqPBAAAAMAntIqQ1K1bt9x8882VHgMAAACA5aj4R9uojOrq6pxyyinLvO4UNDfrj0qx9qgk649KsfaoJOuPSrH2mk9VqVQqVXoIAAAAAFo/ZyQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJn8M777yTrbfeOhdeeGF526233po+ffqkpqam/M9ll11W3v+MM87IAQcckAMPPDA1NTX5n//5n+Ue47//+79z8MEHp6amJvvuu2/OPPPM8n39+vXLs88+22j/Qw45JI888kijbT/60Y+y4447ZvHixeVtDQ0NOeKII3LQQQfloIMOyvHHH59XX321fP8tt9xSnvGQQw7Jn//856VmGzZsWHr06JGFCxcmSR555JFss802qampSW1tbRoaGjJ06NDss88+Oeigg3Lsscfm5ZdfbjT/vvvuW/45/f73vy/f9/bbb+eMM87IPvvsk/333z/jx49PktTV1aWmpia9e/fO/fffv9yfHZ/Pylzfl19+eXbaaaeceuqpSZIXX3wxRx99dPbdd98ccMABGTZsWN5///0kyauvvpotttii0THmz59fnuHvf/97Bg8enP79+6d///558MEHkySPP/54ampqGq1JVl3Nuf6S5Mwzz8zOO++81HpZ3tpMkquuuir9+/fPQQcdlCOPPDLPPfdckg/XbU1NTbbaaqulnpdZtVRq7SXJ5MmTc+CBB2b//ffPSSedlLfffjtJ08+Z1l7LWJVf9yVJjx49ynPU1NTkmWeeSfLha6vjjz8+ffv2Td++fRs95vHHH8/AgQPLf+eOHTs2H11e9aPvfeDAgUmS+fPn58QTT8w+++yTAw88MKecckrmzZvX5PGTD9fxR4/t379/+bc5W98t76PX5wcddFC++93v5uSTT87jjz+eZOU+F37aa/2P++R7Deth1dEc62hFnHfeefmP//iPJvc7+uijs+eee2bChAlJktmzZ+foo4/Odtttl0MOOaTRvvX19RkzZkwOOOCA7LPPPo2eD6dOnZpDDjkkBxxwQPbff//86le/Kj/u7rvvTv/+/Zd6fl3llfjM/uM//qN01FFHlXbcccfSBx98UCqVSqXJkyeXhg4dusz9zz///NLFF19camhoKJVKpdK8efNKr7322qd+/dmzZ5f69u1bev3110ulUqnU0NBQeuqpp8r377HHHqVnnnmm0WMGDBhQevjhh8u358+fX9p+++1Lhx9+eOnuu+9utG9tbW353//93/+99MMf/rA8V+/evUtvvvlmqVQqlaZOnVrab7/9Gj122rRppWHDhpW6d+9eWrBgQalUKpUefvjh0oABA8r71NfXl6ZOnVqqr68vlUql0vXXX1865phjljv/R77//e+XrrvuuvLtOXPmNLr/qKOOKk2fPn2Zj2XlWJnre9y4caWLL764vO8rr7xSXsv19fWl0047rTR+/PjyfTvssMMyj7Fw4cJSv379Sk888USpVCqVFi9eXJo3b16jfT6+Jll1Nef6K5VKpT/+8Y+lt956a6n1sry1OXPmzNLuu+9eWrhwYalUKpUmTpxYOuGEExp93eU9r7FqqNTae/7550s777xzae7cuaVSqVS64oorSj/5yU9KpdLy1+VHrL3mt6q/7vu0vx8XL15ceuihh0ozZ85c6u/fZ555pvTiiy+WSqVS6YMPPigNHDiwdNttty3ze58/f36jWS6++OLSsGHDmjx+Q0NDqaampnTfffeVb7/11luN9rG+W84nf9b33HNPabvttis9+eSTK/W5sKnX+st6r/FpM9L6NMc6WhHnnntu6frrr29yv0++p6ytrS09+uijpfvvv7/R+9pSqVSaNGlS6bjjjivV1dWVFi9eXDr++ONLU6ZMKZVKpdKTTz5ZmjVrVvlr7LXXXqVHH320/Njlvb9ZVTkj6XOYPHlyfvCDH6RHjx6ZNm1ak/vPmjUrG264YaqqqpIka6+9dr761a9+6v5vvfVW2rVrl86dOydJqqqqssUWW6zQjHfeeWd22223DBo0KJMnT25035prrln+9wULFqRNmw+XQ6lUSqlUKtf/d999NxtttFF53/nz52f8+PEZNmzYco/dpk2b7LnnnuWv26tXr7z++utNzvzSSy/l2WefzZAhQ8rb1l9//SYfx8rVnOu7a9eu5bXcpk2bbL311oXWxpQpU7LddtulV69eSZJ27dpl7bXXLvgdsSpp7ufXnXbaKeuuu+5S25e3NquqqrJ48eLymSCffG7ki6FSa+/ZZ59Nz549s8466yRJdtttt9x5551JPvtzJivXqv6679O0a9cu3/72txu9LvxI9+7ds9lmmyVJOnTokC222OJT117nzp0b/R/3oq/7/vjHP6ZTp07Za6+9knz4fS/rvxEqY++9987AgQPzy1/+crn7rch6b+q1ftH3Gqw6VsY6qqury9ixY3PooYfmoIMOytlnn11+vzp79uwMGTIk/fv3z4knntjo0wwLFizI8OHDc+ihh+bAAw/MhRdemPr6+mUef80110yfPn2y+uqrL3Xf008/nZ122int27dPu3bt8p3vfKf89/Q222yTDTfcsPw1unXrltdee20Ff0qrlnaVHmBV9fTTT+ftt9/OjjvumDfffDOTJ0/Ofvvtl+TDvxBramrK+x511FE57LDDcswxx+TUU0/NlClT0rt37/Tr1y877bTTpx5j8803z9Zbb53dd989ffv2zbbbbpuamppGb5xPPfXUrLbaauXbL730UqOvMXny5Jx77rnp1atXxowZk9mzZ5cXeZKceOKJmTlzZtZee+3yf9jrrLNORo8enQEDBqS6ujoNDQ25/vrry48ZPXp0Tj311GW+4FieG264If369Wu07ayzzkqpVMp2222XM844I9XV1Xn++eez4YYbZvjw4fn73/+e9dZbL+ecc06+9a1vrdDx+OxaYn1/5P3338/kyZNzxhlnlLctXLiwfDpp//79c/zxx6eqqirPP/982rVrlxNPPDFz5szJlltumXPPPTdrrbXWSv4JUEktuf6W55Nrc/PNN8+xxx6bfv36Zc0110x1dXWh06ZZdVRy7W2++eb529/+lldeeSVdu3bNlClT8t577+Xtt98uh4Vk2c+ZNL8vyuu+o48+OvX19dl1110zdOjQdOjQofDPYO7cubnnnnvKHwFZnoaGhtx4441Lve5b1vGff/75dO7cOaeeempefvnlfO1rX8uwYcPSpUuXwrPRvLbZZptMnz49u++++0pZ70291v+s7zVo3T7vOrr22muz5ppr5pZbbkmSXHrppZkwYUJOP/30XHjhhdl+++1zyimn5JVXXslBBx2UXXbZJUly0UUXZfvtt8+YMWPS0NCQs846K5MnT87hhx++QvNvueWWufXWWzNo0KAkH36crba2dqn9XnjhhTz55JP56U9/+pl+TqsKIekzuuWWW1JTU5OqqqrsvffeufDCCzN79uwkybe//e2MGzduqcfstNNOuf/++/PII4/ksccey49+9KMcf/zx+Zd/+ZdlHqNNmza58sor8+yzz+bRRx/N1KlT88tf/jJ33nln+UXluHHj0r179/JjPv5ZzpkzZ6a2tjY77rhjec7bb7893//+98v7XHPNNWloaMjVV1+df/u3f8v555+fBQsW5IYbbsgtt9ySb3zjG7nrrrtyyimn5L/+67/y3//932nfvn123333Ffp5XXPNNXnhhRcyceLE8rYbbrghXbp0SV1dXcaMGZPRo0fnsssuS0NDQ/7yl7/kzDPPTJ8+fXLvvffm5JNPztSpU1fomHx2LbG+k2TJkiU5/fTTs+OOO2bPPfdMkmywwQZ58MEHs+6662bu3Lk5+eSTs9Zaa+Wwww5LQ0NDHn744UyaNCnrrbdeLrroolx88cW56KKLmucHQUW01PpbnmWtzddeey3Tpk3Lvffemw022CDXXnttzjvvvFx99dWf/ZulVank2vv617+eESNG5PTTT09VVVV53bVr9/+/VFvWuqRlfBFe9z3wwAPp0qVLFixYkLPPPjtXXHFFTj/99ELf/4IFC3LyySfnuOOOK3SW1AUXXJCOHTvmqKOOKm/7tON/9Hf7TTfdlG7duuW6667Lueeem1//+teFZqP5lf7fdWCSlbPel/da/6677vpM7zVo/T7vOpo+fXoWLFiQe+65J8mHZyhtvvnmST68Vu+IESOSJJtsskmjiDl9+vT89a9/zXXXXZfkw/8h8/HAXtQhhxySV155JUceeWTWWGONbL311nn44Ycb7TNnzpz84Ac/yKhRoz7TMVYlQtJnUFdXlylTpqRDhw654447kiSLFy/Orbfe2uSCWWONNbLnnntmzz33zFZbbZV/+7d/a/LFZvfu3dO9e/fyBYb/9Kc/Ze+9925yzsmTJ6e2trb8YrOuri6dOnVqFJKSD1+4HHroodl7771z/vnn5w9/+EPWXHPNfOMb30jy4Rkhw4YNy/z58/OnP/0pDz/8cKP/w3TAAQfkmmuu+dQ5rr/++kyZMiUTJ05sdJrgR/+nqUOHDhk0aFBOPvnk8vYuXbqkT58+ST48FfLss8/OvHnzyqf803xaan3X19fnrLPOylprrVV+4k8+XA8fndK+7rrr5sADD8zjjz+eww47LF26dEnfvn2zwQYbJEkOPPDA/PjHP14Z3zatREs/vy7Lp63Nu+++O927dy+vv4MPPniZFwdl1dQa1t7++++f/fffP0ny17/+Nb/5zW+yxhprJPn0dUnz+6K87vvoddcaa6yRww47rPymqimLFi3KSSedlO985zs57rjjmtx/7Nix+cc//pGrrrqqfHmD5R2/S5cu2XLLLdOtW7ckyUEHHbTMN5hUzt/+9rdCnwwout6X91p/ee81vvnNb668b4oW93nXUalUyqhRo1b4rN9SqZQrr7wym2yyyWcdPcmH75lPP/30coC/5pprys9byYdnbR577LE54YQTymesfpG5RtJnMG3atHz961/P7373u0yfPj3Tp0/Pr371q9x2223LfdxDDz2UBQsWJPlwQc+cOTNdu3b91P1nz56dJ554onx71qxZmTdv3nIf85GPXvRMnjy5POMf/vCHJMmf//znzJs3r9Fv0rj77rvTo0ePJB9ei2HmzJmZO3dukuThhx/OGmuskbXXXjvnn39+o+87+fC6NZ/2xD5p0qTcdNNNue666xqdmv/ee+/l3XffLf8s7rrrrvTs2TNJstVWW6Vjx47l34b06KOPZq211nItnBbSEuu7oaEh5513Xtq2bZsxY8aUPwedfPgk/NFvmlm0aFGmT59e/r8N++23X/7617+Wj/O73/2uvG75Ymip59dPs7y12bVr1zz22GN57733kiQPPvigj9x+gVR67SXJm2++mST54IMPMm7cuPKb9uWtS5rfF+F13zvvvFO+vtuSJUtyzz33lF93Lc8HH3yQk046Kdtss01OO+20Jvf/+c9/nhkzZuSKK65o9LG55R1/1113zaxZszJnzpwkye9//3t/t7ciU6dOzY033thkRFyR9b681/or+l6DVcPKWEf9+vXLv//7v5efSxYsWJAXXnghSbLjjjuWrwv3yiuvNPpNb/369cuECRPK10WaN29eXnnllRX+Hj744IPy+9fXX389N954Y4499tgkH17X69hjj83gwYNz2GGHrfDXXhU5I+kz+OjX835c796909DQkNdff32pz3xutdVWGTNmTJ555plcfPHF5dP6Nt1004wcOfJTj7NkyZJcfvnlee211/KVr3wlDQ0N+dGPflTolOKpU6fma1/7WjbddNNG2w888MBMnjw5//zP/5xhw4aV37BvvPHGufTSS8vznnDCCTnqqKPSvn37dOjQIb/4xS9W+IXrggULcv755+erX/1q+T+yDh065Oabb87cuXMzdOjQ1NfXp6GhId26dcuoUaOSfHiRxZ/97GcZNmxY6urqsvrqq2f8+PFeOLeQlljfv/vd7/Jf//Vf6d69e/m0/G233TajRo3KY489lnHjxqVNmzZZsmRJdt999/Kp8V/96ldz4oknZuDAgamqqkrXrl1zwQUXNMePgQppqefXU045JX/961+TJPvuu2+6d++eX/7yl8tdm3vvvXf+8pe/5JBDDkmHDh1SXV3tY5VfIJVee8mHv+r69ddfz+LFi9O/f/8cc8wxSZb/nEnz+yK87jv88MMzcuTIVFVVZcmSJendu3ejMPS9730vs2fPTm1tbXbdddfssssuGTNmTG655Zb86U9/yttvv10OU/vuu2/5LPKPe+6553L11Vdns802y8CBA5N8GOCvuOKK/O///u+nHr9jx44ZMWJETjzxxJRKpXTu3DkXX3xxk98zzefUU09Nhw4dsmjRonTr1i0TJkzINttskxdeeGGlrHev9b8cVvY6+pd/+ZeMHz8+hx56aKqqqlJVVZVTTjkl3bp1y/Dhw3POOedkypQp6dq1a6ML///4xz/OpZdeWv54cvv27fPjH/94mWco1dfXZ4899khdXV0WLFiQXXfdNYcddliGDh2ad999N0cffXT5TMuzzjorW265ZZJkwoQJeemll/Kf//mf+c///M8kyTHHHJPvfe97zfPDbQWqSh//sCJ8Do888kjGjh2bW2+9tdmPdfTRR+e4447LHnvs0ezH4vO7/PLL89577+Xcc89t9mP16NEjjz/+eDp16tTsx2LV0JLrr1+/frnqqqsaXcOELy9rjy+yW2+9NQ888ECLfAzN+l61reznQuuBlaWl3lO++uqr+d73vpdHHnmkWY/Tkny0jZWmffv2mTt3bmpqapZ5BfuVoa6uLjU1NXnllVca/dYSWreOHTvmvvvuy6mnntpsx3j88cdTU1OT9dZbr9E1GaAl1t+rr76ampqaLF68uNHFkflys/b4IvvKV76SGTNmlM8+ag7W9xfDynoutB5Y2dZaa61ccsklhX4b5Wd199135+STT856663XbMeoBGcktQLjx4/Pfffdt9T2X/3qV+WLDgMAsOrzug+AVZ2QBAAAAEAhPv8BAAAAQCFCEgAAAACFCEkAAAAAFCIkAQB8ijvvvDOHHHJIevfunZ133jknnHBC/vznP+fyyy/PWWedVd6vR48e6dWrV3r37p1ddtklF110Uerr65N8+Kuq//jHPzb6urfeemuOPPLI8u1+/fpl6623Tu/evcv/jB49urxvz54907t372y77bY56KCDcv/997fAdw8AsDS/NxEAYBmuu+66TJgwIT/96U+z8847p3379vn973+fadOmpWPHjkvtf8cdd2TTTTfNCy+8kGOOOSabbbZZo1jUlKuuuirf/va3l3lfr169cuONN6ahoSE33XRTzjjjjDz44IOprq7+zN8fAMBn4YwkAIBPePfddzNu3LiMHDkye++9dzp27Jj27dunX79+Offcc5f72G7dumW77bbLc889t9LnatOmTWpqavLee+/lpZdeWulfHwCgKUISAMAnPPHEE/nggw/y3e9+d4Uf+/zzz+exxx5Lz549V/pc9fX1ufXWW9O+fftsvPHGK/3rAwA0xUfbAAA+4e23387aa6+ddu2Kv1QaMGBA2rZtm7XWWiuHHnpovve975Xv++EPf5i2bduWby9evDhbbLFFo8d/cp9zzjknhx9+eJLkL3/5S/r06ZNFixalbdu2ueSSS7Luuut+1m8PAOAzE5IAAD6hc+fOmT9/fpYsWVI4Jt12223ZdNNNl3nfFVdc0ej6R7feemtuvvnm5e7zcdtss01uvPHGLFy4MMOHD89jjz2W/v37F/xuAABWHh9tAwD4hN69e6dDhw6ZOnVqpUdppFOnTjn//PNzxx13ZObMmZUeBwD4EhKSAAA+Yc0118ypp56a0aNHZ+rUqVm0aFEWL16cBx98MJdccklFZ+vcuXMOO+ywXHHFFRWdAwD4cvLRNgCAZTjuuOOy3nrr5corr8xZZ52VTp06Zcstt8xJJ52Uhx56aKUf76STTmp0jaRvf/vbnxqLhgwZkr322itPP/10Nt9885U+CwDAp6kqlUqlSg8BAAAAQOvno20AAAAAFCIkAQAAAFCIkAQAAABAIUISAAAAAIUISQAAAAAUIiQBAAAAUIiQBAAAAEAhQhIAAAAAhQhJAAAAABTy/wEZ5wVAXKvHFgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x=\"CIPHER\",y=\"TASA(MB/SEC)\",data=_info_df)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
