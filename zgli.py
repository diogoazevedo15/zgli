#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

    
######################################################################### CATEGORIZE_COLS ###########################################################################
def categorize_cols(df,cols,cuts):
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
    standardize_categorical_cols : Standardize the given categorical columns into the same format.
    encode_df: Encode the given categorical columns into patterned strings.
    
    Notes
    -----
    This function uses the pd.cut function from the pandas library to categorize the given columns.
    
    Examples
    --------
    # Imports
    >>> import zgli
    >>> from sklearn import datasets
    
    # Load Iris df
    >>> iris = datasets.load_iris()
    >>> iris_df = pd.DataFrame(iris['data'])

    # Encode iris df
    >>> cols = [0,1,2,3]

    # Divide iris df
    >>> cuts = [4,4,4,4]
    >>> df_ct = categorize_cols(iris_df,cols,cuts) # We use the categorize function here.
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

def standardize_categorical_cols(df,cols):
    
    """
    Standardize the given categorical columns into the same format.
    
    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe with categorical columns we wish to standardize.
        
    cols : array_like
        A list with the columns to be standardized.
        
    Returns
    -------
    df_st : pandas.DataFrame
        The Dataframe with the given columns standardized.

    See Also
    --------
    categorize_cols : Divide continuous columns into categories.
    encode_df: Encode the given categorical columns into patterned strings.
    
    Notes
    -----
    This function takes categorical columns and turns categories into a number sequence i.e [0,1[ becomes 0, [1,2[ becomes 1, etc... 
    This happens to each column separately, so the same category [1,2[ of two different columns can be encoded as a 0 in the fisrt column, if
    this is the first category in the column and as a 1 in the seccond column if it was the seccond category found in the column.
    
    Examples
    --------
    # Imports
    >>> import zgli
    >>> from sklearn import datasets
    
    # Load Iris df
    >>> iris = datasets.load_iris()
    >>> iris_df = pd.DataFrame(iris['data'])

    # Encode iris df
    >>> cols = [0,1,2,3]

    # Divide iris df
    >>> cuts = [4,4,4,4]
    >>> df_ct = categorize_cols(iris_df,cols,cuts)
    >>> df_ct.head()
    		0		1		2		3
    0	(4.296, 5.2]	(3.2, 3.8]	(0.994, 2.475]	(0.0976, 0.7]
    1	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
    2	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
    3	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
    4	(4.296, 5.2]	(3.2, 3.8]	(0.994, 2.475]	(0.0976, 0.7]
    
    # Standardize df_div iris df
    >>> df_std = standardize_categorical_cols(df_ct,cols) # We use the standardize function here.
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
def encode_df(df,cols,hop=1):

    """
    Encode the given categorical columns into patterned strings.
    
    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe with standardized categorical columns obtained using the standardize_categorical_cols function.
        
    cols : array_like
        A list with the columns to be encoded.
        
    hop : int
        A number representing the the character difference between each patterned string. Ex: hop = 1 s1 = 000000 s2 = 010101 | hop = 2 s1 = 000000 s2 = 012012
        
    Returns
    -------
    files : pandas.DataFrame
        The Dataframe with the given columns encoded.

    See Also
    --------
    categorize_cols : Divide continuous columns into categories.
    encode_df: Encode the given categorical columns into patterned strings.
    
    Notes
    -----
    This function takes standardized categorical columns and turns categories into patterned strings taking into account the hop given.
    
    Examples
    --------
    # Imports
    >>> import zgli
    >>> from sklearn import datasets
    
    # Load Iris df
    >>> iris = datasets.load_iris()
    >>> iris_df = pd.DataFrame(iris['data'])

    # Encode iris df
    >>> cols = [0,1,2,3]

    # Divide iris df
    >>> cuts = [4,4,4,4]
    >>> df_ct = categorize_cols(iris_df,cols,cuts)
    >>> df_ct.head()
    		0		1		2		3
    0	(4.296, 5.2]	(3.2, 3.8]	(0.994, 2.475]	(0.0976, 0.7]
    1	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
    2	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
    3	(4.296, 5.2]	(2.6, 3.2]	(0.994, 2.475]	(0.0976, 0.7]
    4	(4.296, 5.2]	(3.2, 3.8]	(0.994, 2.475]	(0.0976, 0.7]
    
    # Standardize df_div iris df
    >>> df_std = standardize_categorical_cols(df_ct,cols)
    >>> df_std.head()
    0	1	2	3
    0	0	2	0	0
    1	0	1	0	0
    2	0	1	0	0
    3	0	1	0	0
    4	0	2	0	0
    
    # Encode df
    >>> hop = 1
    >>> df_enc = encode_df(df_std,cols,hop) # We use the encoding function here.
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
        df_aux[col] = df_aux[col].apply(lambda x: sequences[x])

    return df_aux

########################################################################## GENERATE FILES #########################################################################
def generate_files(df,file_cols, name_cols, out_path, sep = '', verbose = 1):
    
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
            file_content = ''.join(row[1][cols].values)
        else:
            file_content = ''.join(row[1][cols].values + sep)

        file_path = os.path.join(out_path,name)
        with open(file_path, 'w') as f:
            f.write(file_content)
            
        if verbose == 1:
            print('######################')
            print('File: ', name)
            print(file_content)
    
    print('\nDone.')
        

############ ZGLI ############
class Folder:
       
    class Raw:
         def compress(self,col):
            return col
  
    ################################################################################ INNIT ################################################################################
    def __init__(self,folder_path):
        
        """
        Folder
        
        Perform operations inside the folder containing the files intended for clustering
        
        Parameters
        ----------
        folder_path : string
            The folder path where all the files to clustered are.
            
        See Also
        --------
        folder.distance_matrix : Compute a distance matrix of all files using the normalized compression distance.
        folder.get_file_names: Return the names of all the files inside the folder.
        
        Notes
        -----
        This class is intended to represent the folder containing the files to be clustered.
        Beyond performing the ncd between all files it also provides some functions related to the
        files to further simplify some operations while manipulating them.
        
        Examples
        --------
        # Imports
        >>> from zgli import Folder
        
        # Initialize class
        >>> data_path = 'D:/folder'
        >>> folder = Folder(data_path)
        >>> folder
        <zgli.Folder at 0x21e02eb4c40>
            
        """
        
        # Initialize folder path (with all files to cluster).
        # Get the file names.
        # Initialize Raw() aux class, to get results without compression.
        
        self.files_path = folder_path
        self.file_names = next(os.walk(self.files_path), (None, None, []))[2]  # [] if no file
        self.raw = self.Raw()
    
    ################################################################################ FILE MANAGEMENT #######################################################################
    def get_file_names(self):
        
        """
        Return a list with the name of all text files inside the folder.
        
        Parameters
        ----------
        This function has no parameters.
        
        Returns
        -------
        file_names : list()
            The name name of all files text inside the folder.
            
        See Also
        --------
        folder.get_file_lengths : Compute a distance matrix of all files using the normalized compression distance.
        folder.get_file_sizes: Return the names of all the files inside the folder.
        
        Examples
        --------
        # Imports
        >>> from zgli import Folder
        
        # Initialize class
        >>> data_path = 'D:/folder'
        >>> folder = Folder(data_path)
        >>> file_names = folder.get_file_names()
        >>> file_names
        ['blueWhale.txt',
         'cat.txt',
         'chimpanzee.txt',
         'finWhale.txt',
         'graySeal.txt',
         'harborSeal.txt',
         'horse.txt',
         'human.txt',
         'mouse.txt',
         'rat.txt']
        """
        
        # Simply return file names that were obtained during init.
        
        return self.file_names
        
    def get_file_lengths(self):
        
        """
        Return a dicionary with the name of all files text inside the folder algongside their length i.e their number of rows.
        
        Parameters
        ----------
        This function has no parameters.
        
        Returns
        -------
        file_lengths : dict{}
            The name of all files text inside the folder algongside their length i.e their number of rows.
            
        See Also
        --------
        folder.get_file_names: Return a list with the name of all text files inside the folder.
        folder.get_file_sizes: Return the names of all the files inside the folder.
        
        Examples
        --------
        # Imports
        >>> from zgli import Folder
        
        # Initialize class
        >>> data_path = 'D:/folder'
        >>> folder = Folder(data_path)
        >>> file_lenghts = folder.get_file_lenghts()
        >>> file_lenghts
        {'files': ['blueWhale.txt',
         'cat.txt',
         'chimpanzee.txt',
         'finWhale.txt',
         'graySeal.txt',
         'harborSeal.txt',
         'horse.txt',
         'human.txt',
         'mouse.txt',
         'rat.txt'],
         'lenght': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
         
        """
        
        # Return the files length in rows. Output comes in a dictionary {'files': file_names, 'lenght': files_len }.
        # Read files with 'latin-1' encoding so we avoid missing characters for Portuguese files.
        # Use with open() so the file closes automatically.
        files_len =[]

        for name in self.file_names:
            with open(os.path.join(self.files_path,name), encoding='latin-1') as fp:
                file = fp.read()
                file_split = file.split('\n')
                files_len.append(len(file_split))

        return {'files': self.file_names, 'lenght': files_len}
    
    def get_file_sizes(self):
        
        """
        Return a dictionary with the name of all files text inside the folder algongside their size i.e their number of characters.
        
        Parameters
        ----------
        This function has no parameters.
        
        Returns
        -------
        file_lengths : dict{}
            The name name of all files text inside the folder algongside their length i.e their number of rows.
            
        See Also
        --------
        folder.get_file_names: Return a list with the name of all text files inside the folder.
        folder.get_file_lengths: Return a dicionary with the name of all files text inside the folder algongside their length i.e their number of rows.
        
        Examples
        --------
        # Imports
        >>> from zgli import Folder
        
        # Initialize class
        >>> data_path = 'D:/folder'
        >>> folder = Folder(data_path)
        >>> file_sizes = folder.get_file_sizes()
        >>> file_sizes
        {'files': ['blueWhale.txt',
          'cat.txt',
          'chimpanzee.txt',
          'finWhale.txt',
          'graySeal.txt',
          'harborSeal.txt',
          'horse.txt',
          'human.txt',
          'mouse.txt',
          'rat.txt'],
         'size': [16440,
          17040,
          16620,
          16440,
          16800,
          16860,
          16680,
          16620,
          16320,
          16320]}
        """
        
        # Return the files size as a the number of characters inside the file. Output comes in a dictionary {'files': self.file_names, 'size': files_len}.
        # Read files with 'latin-1' encoding so we avoid missing characters for Portuguese files.
        # Use with open() so the file closes automatically.
        
        files_len =[]

        for name in self.file_names:
            with open(os.path.join(self.files_path,name), encoding='latin-1') as fp:
                file = fp.read()
                files_len.append(len(file))

        return {'files': self.file_names, 'size': files_len}
    
    def get_files_content(self,by_column=0,delimiter=','):
        
        """
        # Return the files content. Output comes in a dictionary {'files': self.file_names, 'content': files_content}. 
        # Read files with 'latin-1' encoding so we avoid missing characters for Portuguese files.
        # Use with open() so the file closes automatically.
        """
        
        files_content =[]
        for name in self.file_names:
            path = os.path.join(self.files_path,name)
            
            # Read the
            if by_column == 0: 
                with open(path, encoding='latin-1') as fp:
                    file = fp.read()
                    files_content.append(file)
            else:
                files_content.append(np.loadtxt(path, dtype=str, delimiter=delimiter).T)

        return {'files': self.file_names, 'content': files_content}
    
    def clear_folder(self):
        
        """
        Delete all text files insde the folder path.
        
        Raises
        ------
        Exception
            If it fails to delete a file.
            
        See Also
        --------
        genreate_files:  Generate text files given a dataframe.
        folder.get_file_lengths: Return a dicionary with the name of all files text inside the folder algongside their length i.e their number of rows.
        
        Examples
        --------
        # Imports
        >>> from zgli import Folder
        
        # Initialize class
        >>> data_path = 'D:/folder'
        >>> folder = Folder('D:/output/')
        >>> folder.clear_folder()
        Deleting files...

        All files deleted.
       
        """
        
        folder = self.files_path
        print('Deleting files...')
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        
        print('\nAll files deleted.')
    
    ################################################################################ NCD ################################################################################
    
    ###### NORMAL COMPRESISON ######
    def normal_compression(self,compressor,compressors):
        
        """
        # Performs normal copression i.e compresses all the file content together, without taking into account a possible columnar structure.
        # Outputs a matrix with 1st row having the files compressed sizes, and the other rows having the appended files compressed size.
        """
        
        # Get files content
        files_content = self.get_files_content()
        
        # Initialize matrix
        compressed_matrix = []

        # Compress files (p.e K(file1))
        compressed_matrix.append([len(compressors[compressor].compress((content).encode('latin-1'))) for content in files_content['content']])

        # Compress appended files (p.e K(file1+file2))
        for i,content_i in enumerate(files_content['content']):
            row = []
            for j,content_j in enumerate(files_content['content']):
                if i == j:
                    row.append(0)
                else:
                    row.append(len(compressors[compressor].compress((content_i+content_j).encode('latin-1'))))
            compressed_matrix.append(row)
            
        return compressed_matrix
    
    ###### COMPRESSION BY COLUMN ######
    def compression_by_column(self, compressor, compressors, delimiter, weights):
        
        """
        # Performs compression by collumn i.e compresses each column in the files seperatly, and sums the column compressed sizes to define the final file compressed size.
        # Compression of appended files are made column by column as well, where file1_col1 is appended to file2_col1 and compressed. This repeates until all all columns 
        # have been appended and compressed. The sum of all appended and compressed files defines the final appended files size.
        # Outputs a matrix with 1st row having the files compressed sizes, and the other rows having the appended files compressed size.
        """
        
        # Get files content
        files_content = self.get_files_content(by_column = 1, delimiter = delimiter)
        
        # Initialize matrix
        compressed_matrix = []

        # Check if number of columns and number of compressors match.
        if isinstance(compressor,list) and (len(files_content['content'][0]) != len(compressor)):
            raise Exception("Number of columns and compressors do not match")
            
        # Generate weights if there are None. Generates array of 1 of lenght equal to the number of columns
        if weights == None:
            weights = [1] * len(files_content['content'][0])

        # Compress files (K(file1))
        row = []
        for content in files_content['content']:
            file_length = 0
            for col_i,col in enumerate(content):
                
                # If compressor is a sigle string, use it to compress all the columns.
                if  isinstance(compressor,str):
                    file_length += (len(compressors[compressor].compress(np.ascontiguousarray(col))) * weights[col_i])
                    
                # If compressor is a list of size equal to the number of columns, use same index compressors to compress same index columns.
                elif  isinstance(compressor[col_i],str):
                    file_length += (len(compressors[compressor[col_i]].compress(np.ascontiguousarray(col)))* weights[col_i])
            
            # Append file compressed size to row
            row.append(file_length)
        
        # Append first row containing compressed file sizes to matrix
        compressed_matrix.append(row)

        # Compress appended files (p.e K(file1+file2))
        for i,content_i in enumerate(files_content['content']):
            row = []
            for j,content_j in enumerate(files_content['content']):
                if i == j:
                    row.append(0)
                else:
                    
                    # Load files with only one row.
                    # A different way was implemented sice np.hstack did not behave the way we wanted when the files had a single row.
                    if content_i.ndim == 1:
                        content_aux = np.core.defchararray.add(content_i, content_j)
                        content = [ [k] for k in content_aux]

                    # Load files with more than a sigle row.
                    # np.hstack, appends the files columns together.
                    else:
                        content = np.hstack((content_i,content_j))

                    # Compress appended columns
                    file_length = 0
                    for col_i,col in enumerate(content):
                        
                        # If compressor is a sigle string, use it to compress all the columns.
                        if  isinstance(compressor,str):
                            file_length += (len(compressors[compressor].compress(np.ascontiguousarray(col)))* weights[col_i])
                            
                        # If compressor is a list of size equal to the number of columns, use same index compressors to compress same index columns.
                        elif  isinstance(compressor[col_i],str):
                            file_length += (len(compressors[compressor[col_i]].compress(np.ascontiguousarray(col)))* weights[col_i])
                    
                    # Append file lenght to matrix row
                    row.append(file_length)
                    
            # Append row containing appended compressed file sizes to matrix
            compressed_matrix.append(row)
        return compressed_matrix
    
        
    ###### COMPUTE PAIR DISTANCES ######
    
    # GET COMPRESSED SIZES
    def get_compressed_sizes(self,compressor, compress_by_col, delimiter, weights):
        
        # Define compressors. NEW COMPRESSORS SHOULD BE ADDED HERE
        compressors = {
            'bzlib':bz2, 
            'zlib':zlib, 
            'lzma':lzma, 
            'gzip':gzip, 
            'raw':self.raw
        }
        
        # Initialize matrix
        compressed_matrix = []
        
        # Normal Compression
        if compress_by_col == 0:
            compressed_matrix = self.normal_compression(compressor,compressors)
            
        # By column: one compressor for all columns
        elif compress_by_col == 1:
            compressed_matrix = self.compression_by_column(compressor, compressors, delimiter, weights)
              
        return compressed_matrix
    
    # COMPUTE UNCOMPRESSED MATRIX
    def get_uncompressed_sizes(self):
        file_sizes = self.get_file_sizes()['size']
        uncompressed_matrix = [file_sizes]
        for i,file_1 in enumerate(file_sizes):
            row = []
            for j,file_2 in enumerate(file_sizes):
                if i == j:
                    row.append(0)
                else:
                    row.append(file_1 + file_2)
            uncompressed_matrix.append(row)
        return(uncompressed_matrix)
    
    # COMPUTE NCD
    def distance_matrix(self, compressor, output_path = None, compress_by_col = 0, delimiter = ',', weights = None, verbose = 1):
        
        """
        Computes the normalized compression distance between all documents inside the folder.
        
        Parameters
        ----------
        compressor : {'zlib','gzip','bzlib','lzma','raw'}
            Wich compressor to use. Diferent compressors may yield diferent results when clustering.
            
            - 'zlib' produces asymmetrical matrices and smaller sizes for small strings.
            - 'gzip'  has simillar behavior to zlib (normally with bigger compressed sizes).
            - 'bzlib' produces symmetrical matrices and is recomended when using data that was encoded using the zgli.encode_df function.
            - 'lzma' usually produces the most compressed sizes.
            
        output_path : string, default=None 
            The file path of the to where the distance matrix should be outputed.
            
        compress_by_col : bool, default=False 
            Defines if the data shoul be compressed by column or normally. Default = False, i.e the files are compressed normally.
        
        delimiter : string, default=',' 
            The character to be used to separete columns, default = ',' since .csv is a common format for tabular data.
            
        weights : list, default=None
            A product between weight[i] and column[i] is computed if weights are provided. Default is none i.e all columns have weight = 1.
            
        verbose : int, default=1 
            Controls verbosity when generating files. Default is 1 so the distance matrix is shown.
        
        Returns
        -------
        distance_matrix : list(list())
            All the ncds between the files inside the folder
            
        Outputs
        -------
        distance_matrix : .txt
            A .txt containing a matrix of the same format as the one printed to the screen when verbose = 1
        
            
        See Also
        --------
        genreate_files:  Generate text files given a dataframe.
        folder.get_file_lengths: Return a dicionary with the name of all files text inside the folder algongside their length i.e their number of rows.
        
        Examples
        --------
        # Imports
        >>> from zgli import Folder
        
        # Define Parameters
        data_path = '../../data/examples/10-mammals'
        compressor = 'bzlib'
        output_path = 'D:/Fcul/Tese/DockerFolder/'
        
        # Initialize Folder class
        folder = Folder(data_path)
        
        # Compute matrix
        dm = folder.distance_matrix(compressor, output_path)
        0_mouse.txt 0.0 0.941648 0.964551 0.967002 0.957282 0.960252 0.960088 0.967124 0.960965 0.965251
        0_rat.txt 0.941648 0.0 0.966302 0.96132 0.958167 0.958732 0.966924 0.960157 0.955044 0.958172
        1_graySeal.txt 0.964551 0.966302 0.0 0.77382 0.96105 0.959818 0.954923 0.964729 0.949891 0.942299
        1_harborSeal.txt 0.967002 0.96132 0.77382 0.0 0.960009 0.960252 0.953671 0.962769 0.947334 0.94423
        2_blueWhale.txt 0.957282 0.958167 0.96105 0.960009 0.0 0.955691 0.854465 0.960157 0.949342 0.95903
        2_chimpanzee.txt 0.960252 0.958732 0.959818 0.960252 0.955691 0.0 0.953301 0.8649 0.949175 0.961604
        2_finWhale.txt 0.960088 0.966924 0.954923 0.953671 0.854465 0.953301 0.0 0.956238 0.948465 0.958172
        2_human.txt 0.967124 0.960157 0.964729 0.962769 0.960157 0.8649 0.956238 0.0 0.95123 0.959459
        4_horse.txt 0.960965 0.955044 0.949891 0.947334 0.949342 0.949175 0.948465 0.95123 0.0 0.952166
        5_cat.txt 0.965251 0.958172 0.942299 0.94423 0.95903 0.961604 0.958172 0.959459 0.952166 0.0
        
        """
        
        # Check if there were given multiple compressors for normal compression.
        if compress_by_col == 0 and not isinstance(compressor,str) :
            raise Warning("Multiple compressors given for simple compression. If compression by column is wanted use compressed_by_col = True")
            
        # Check if there were weights given for normal compression.
        if compress_by_col == 0 and weights != None :
            raise Warning("Weights were given for normal compression. If you wish to use weights please use them in conjunction with compression_by_col = 1")
            
        # Get compressed size matrix
        uncompressed_matrix = self.get_uncompressed_sizes()
        compressed_matrix = self.get_compressed_sizes(compressor, compress_by_col, delimiter, weights)
            
        # Get file sizes and appended file sizes to compute the NCD
        file_sizes = compressed_matrix[0]
        append_file_sizes = compressed_matrix[1:]
        
        # Initialize data string, to be used as file content if necessary
        data = ''
        distance_matrix =  []
        for i, k1 in enumerate(file_sizes):
            row = []
            data += self.file_names[i]
            for j, k2 in enumerate(file_sizes):
                
                # Do not compute distance between the same file
                if i == j:
                    dist = 0.0
                
                else:
                    # COMPUTE DESIRED DISTANCE
                    dist = (append_file_sizes[i][j] - min([k1,k2])) / max([k1,k2])
                    
                # Append to distmatrix.txt data string
                data = data + ' ' + str(round(dist,6))
                row.append(round(dist,6))        
            data += '\n'
            distance_matrix.append(row)
        
        # Crete distance matrix file
        if output_path != None:
            f = open(os.path.join(os.path.join(output_path,'distmatrix.txt')), "w")
            f.write(data)
            f.close()
        
        # Verbose if conditions
        if verbose == 1:
            print(data)
            
        if verbose == 2:  
            maxi = max((np.asarray(distance_matrix)).flatten())
            mini = sorted((np.asarray(distance_matrix)).flatten())[len(distance_matrix)]
            amp = maxi-mini
            print('Max: ', round(maxi,6))
            print('Min: ', round(mini,6))
            print('Amp: ', round(amp,6))

        if verbose == 3:
            print(data)
            
            maxi = max((np.asarray(distance_matrix)).flatten())
            mini = sorted((np.asarray(distance_matrix)).flatten())[len(distance_matrix)]
            amp = maxi-mini
            print('Max: ', round(maxi,6))
            print('Min: ', round(mini,6))
            print('Amp: ', round(amp,6))
            
        return distance_matrix


# In[ ]:




