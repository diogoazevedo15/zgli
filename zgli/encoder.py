############ IMPORTS ############
# OS
import os
import shutil

# Compressors
import gzip
import bz2
import zlib # Text compressor
import lzma # Text compressor

# Aux
import numpy as np
import pandas as pd
import math
from tqdm.notebook import tqdm

# Paralel processes
from multiprocess import Pool

############ ENCODERS ############
class Encoder:

    """
    Encode tabular data using the functions the Encoder class provides.

    See Also
    --------
    encoder.categorize_cols : Divide continuous columns into categories.
    encoder.standardize_categorical_cols : Standardize the given categorical columns into the same format.
    encoder.encode_df: Encode the given categorical columns into patterned strings.

    Examples
    --------
    # Imports
    >>> from zgli import Encoder
    >>> encoder = Encoder()
    >>> encoder
    <__main__.Encoder at 0x212d015c340>
    """

    def perc_arr(self,a):
        perc_arr = [0]

        hop = 100 / len(a[1:])
        current = hop
        for i in a[1:]:
            perc_arr.append(round(current,3))
            current += hop

        return perc_arr

    def get_seq(self,a,b,pos):

        ap = self.perc_arr(a)
        bp = self.perc_arr(b)

        # if percentiles of a and b match
        if bp[pos] in ap:

            return a[ap.index(bp[pos])]

        # if percentiles of a and be do not match
        else:

            # compute all abs values of a - b[pos].
            # Get min(abs_values) i.e the smaller difference between b[pos] and all of a
            # Get the index of min(abs_value) i.e the closest matching percentile between b[pos] and the ap array.
            # We can use this index to then get the sequence corresponding to the b[pos] (either the exact one or an aproximated one).

            abs_values = []
            for i in ap:
                abs_values.append(abs(i - bp[pos]))

            index = abs_values.index(min(abs_values))
            return a[index]

    def categorize_cols(self,df,cols,cuts):
        """
        Divide continuous columns into categories.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe with discrete/continuous columns we wish to categorize.

        cols : array_like
            A list with the columns to be categorized.

        cuts : array_like
            A list of length equal to the number of columns given in cols, containing the number of cuts to be done to each column.

        Returns
        -------
        df_ct : pandas.DataFrame
            The Dataframe with the given columns divided in the number of categories given.

        See Also
        --------
        encoder.standardize_categorical_cols : Standardize the given categorical columns into the same format.
        encoder.encode_df: Encode the given categorical columns into patterned strings.

        Notes
        -----
        This function uses the pd.cut function from the pandas library to categorize the given columns.

        Examples
        --------
        # Imports
        >>> from zgli import Encoder
        >>> from sklearn import datasets

        # Load Iris df
        >>> iris = datasets.load_iris()
        >>> iris_df = pd.DataFrame(iris['data'])

        # Define iris df cols
        >>> cols = [0,1,2,3]

        # Divide iris df
        >>> cuts = [4,4,4,4]
        >>> encoder = Encoder()
        >>> df_ct = encoder.categorize_cols(iris_df,cols,cuts) # We use the categorize function here.
        >>> df_ct.head()
                0		1		2		3
        0	(4.296, 5.2]	(3.2, 3.8]	(0.994, 2.475]	(0.0976, 0.7]
        1	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
        2	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
        3	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
        4	(4.296, 5.2]	(3.2, 3.8]	(0.994, 2.475]	(0.0976, 0.7]

        """

        df_aux = df.copy()

        for i,col in enumerate(cols):
            df_aux[col] = pd.cut(df_aux[col],cuts[i])

        return df_aux

    def standardize_categorical_cols(self,df,cols):

        """
        Standardize the given categorical columns into the same format.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe with categorical columns we wish to standardize.

        cols : list
            A list with the columns to be standardized.

        Returns
        -------
        df_st : pandas.DataFrame
            The Dataframe with the given columns standardized.

        See Also
        --------
        encoder.categorize_cols : Divide continuous columns into categories.
        encoder.encode_df: Encode the given categorical columns into patterned strings.

        Notes
        -----
        This function takes categorical columns and turns categories into a number sequence i.e [0,1[ becomes 0, [1,2[ becomes 1, etc... 
        This happens to each column separately, so the same category [1,2[ of two different columns can be encoded as a 0 in the fisrt column, if
        this is the first category in the column and as a 1 in the seccond column if it was the seccond category found in the column.

        Examples
        --------
        # Imports
        >>> from zgli import Encoder
        >>> from sklearn import datasets

        # Load Iris df
        >>> iris = datasets.load_iris()
        >>> iris_df = pd.DataFrame(iris['data'])

        # Define iris df cols
        >>> cols = [0,1,2,3]

        # Divide iris df
        >>> cuts = [4,4,4,4]
        >>> encoder = Encoder()
        >>> df_ct = encoder.categorize_cols(iris_df,cols,cuts)
        >>> df_ct.head()
                0		1		2		3
        0	(4.296, 5.2]	(3.2, 3.8]	(0.994, 2.475]	(0.0976, 0.7]
        1	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
        2	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
        3	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
        4	(4.296, 5.2]	(3.2, 3.8]	(0.994, 2.475]	(0.0976, 0.7]

        # Standardize df_div iris df
        >>> df_std = encoder.standardize_categorical_cols(df_ct,cols) # We use the standardize function here.
        >>> df_std.head()
        0	1	2	3
        0	0	2	0	0
        1	0	1	0	0
        2	0	1	0	0
        3	0	1	0	0
        4	0	2	0	0

        """

        df_aux = df.copy()

        for i,col in enumerate(cols):

            # Get unique values of current column
            # Sort them from smaller to largest/alphabetically
            # Conver to list to have access to .index function
            unique_values = np.array(df_aux[col].unique())
            unique_values = np.sort(unique_values)
            unique_values = list(unique_values)

            # Replace category by the index its index in the unique_values list
            df_aux[col] = df_aux.apply(lambda row : unique_values.index(row[col]), axis = 1)

        return df_aux

    ############################################################################ ENCODE DF ############################################################################
    def encode_df(self,df,cols,hop=1):

        """
        Encode the given categorical columns into patterned strings.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe with standardized categorical columns obtained using the standardize_categorical_cols function.

        cols : list
            A list with the columns to be encoded.

        hop : int
            A number representing the the character difference between each patterned string. Ex: hop = 1 s1 = 000000 s2 = 010101 | hop = 2 s1 = 000000 s2 = 012012

        Returns
        -------
        df_enc : pandas.DataFrame
            The Dataframe with the given columns encoded.

        See Also
        --------
        encoder.categorize_cols : Divide continuous columns into categories.
        standardize_categorical_cols: Standardize the given categorical columns into the same format.

        Notes
        -----
        This function takes standardized categorical columns and turns categories into patterned strings taking into account the hop given.

        Examples
        --------
        # Imports
        >>> from zgli import Encoder
        >>> from sklearn import datasets

        # Load Iris df
        >>> iris = datasets.load_iris()
        >>> iris_df = pd.DataFrame(iris['data'])

        # Encode iris df
        >>> cols = [0,1,2,3]

        # Divide iris df
        >>> cuts = [4,4,4,4]
        >>> encoder = Encoder()
        >>> df_ct = encoder.categorize_cols(iris_df,cols,cuts)
        >>> df_ct.head()
                0		1		2		3
        0	(4.296, 5.2]	(3.2, 3.8]	(0.994, 2.475]	(0.0976, 0.7]
        1	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
        2	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
        3	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
        4	(4.296, 5.2]	(3.2, 3.8]	(0.994, 2.475]	(0.0976, 0.7]

        # Standardize df_div iris df
        >>> df_std = encoder.standardize_categorical_cols(df_ct,cols)
        >>> df_std.head()
        0	1	2	3
        0	0	2	0	0
        1	0	1	0	0
        2	0	1	0	0
        3	0	1	0	0
        4	0	2	0	0

        # Encode df
        >>> hop = 1
        >>> df_enc = encoder.encode_df(df_std,cols,hop) # We use the encoding function here.
        >>> df_enc.head()
                0		1		2		3
        0	000000000000	012012012012	000000000000	000000000000
        1	000000000000	010101010101	000000000000	000000000000
        2	000000000000	010101010101	000000000000	000000000000
        3	000000000000	010101010101	000000000000	000000000000
        4	000000000000	012012012012	000000000000	000000000000

        """

        # Initialize vars
        df_aux = df.copy()
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        n_classes = [df_aux[col].nunique() for col in cols]

        # Error check
        if max(n_classes) * hop > len(alphabet):
            raise Exception("Number of classes * hop excedes alphabet size. Try reducing number of classes or hop value.")

        # Define slice of alphabet to use. Get arr positions to calculate lcm to get size of strings.
        alphabet_to_use = alphabet[:max(n_classes) * hop]
        arr_positions = np.array(range(0,len(alphabet_to_use)+1,hop))
        lcm = np.lcm.reduce(arr_positions[1:])
        print('SEQUENCE SIZE: ',lcm)

        # Generate string sequences (p.e 'abababababababab')
        sequences = []
        for position in arr_positions:
            sequence = ''
            while len(sequence) < lcm:
                 sequence += alphabet_to_use[:position+1]
            sequences.append(sequence)

        # Encode dataframe
        for col in cols:

            a = np.arange(0,max(n_classes))
            b = np.sort(df_aux[col].unique())

            df_aux[col] = df_aux[col].apply(lambda pos: sequences[self.get_seq(a,b,pos)])

        return df_aux

    ########################################################################## GENERATE FILES #########################################################################
    def generate_files(self,df,file_cols, name_cols, out_path, sep = '', verbose = 1):

        """
        Generate text files given a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe with the content we intend to use to generate our files.

        file_cols : array_like
            A list with the columns to be used as the files content.

        name_cols : array_like
            A list with the columns to be used as the file name Ex: col1 = Blop col2= Guy then, file_name = Blop_Guy.txt.

        out_path : string
            The file path of the folder to where the generated files should be outputed to.

        sep : string, default='' 
            The character to be used to separete columns, default = '' i.e no character is introduced to separate the columns.

        verbose : int, default=1 
            Controls verbosity when generating files. Default is 1 so file name and file content are shown.

        Outputs
        -------
        files : .txt
            Text files containing the content of the 'file_cols' and named after the content inside the 'name_cols'.

        See Also
        --------
        categorize_cols : Divide continuous columns into categories.
        encode_df: Encode the given categorical columns into patterned strings.

        Notes
        -----
        This function may have issues generating the files if the the user does not have access permission to the out_path provided.
        If there is trouble with the finding/accessing the out_path, try changing it to something the user has access for sure.

        Examples
        --------
        # Example Dataframe
        >>> d = {'col1': ['First', 'Second'], 'col2': ['File', 'File'], 'col3': [':)', ':D']}
        >>> df = pd.DataFrame(data=d)

        # Define parameters
        >>> file_cols = ['col1','col2','col3']
        >>> name_cols = ['col1','col2']
        >>> out_path = 'D:/output/'
        >>> sep = ' '

        # Generate files
        >>> generate_files(df,file_cols, name_cols, out_path, sep) # We use the generate funtion here
        Generating files...
        ######################
        File:  First_File.txt
        First File :) 
        ######################
        File:  Second_File.txt
        Second File :D 

        Done.

        """

        cols = file_cols
        print('Generating files...')
        for row in df.iterrows():

            name = ''
            for n in name_cols:
                name = name + str(row[1][n]) + '_'
            name = name[:len(name) -1] + '.txt'

            if sep == '':
                file_content = [str(i) for i in row[1][cols].values]
                file_content_str = ''.join(file_content)
            else:
                file_content = [str(i) + sep for i in row[1][cols].values]
                file_content_str = ''.join(file_content)
                file_content_str = file_content_str[:len(file_content_str)-1]

            file_path = os.path.join(out_path,name)
            with open(file_path, 'w') as f:
                f.write(file_content_str[:-1])

            if verbose == 1:
                print('######################')
                print('File: ', name)
                print(file_content)

        print('\nDone.')