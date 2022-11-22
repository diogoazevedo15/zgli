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

############ FOLDER ############
class Folder:

    class Raw:
         def compress(self,col):
            return col

    ################################################################################ INNIT ################################################################################
    def __init__(self,folder_path):

        """
        Folder

        Perform operations inside the folder containing the files intended for clustering.

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
        file_sizes : dict{}
            The name name of all files text inside the folder algongside their size i.e their number of characters.

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

    ###### COMPRESS FILES######
    def compress_files(self,inp):

        file_content = inp[0]
        compressor = inp[1]

        return len(compressor.compress((file_content).encode('latin-1')))

    ###### COMPRESS APPENDED FILES######
    def compress_appended_files(self,inp):

        file_content = inp[0]
        files_content = inp[1]
        compressor = inp[2]

        row = []
        for content_j in files_content:
            if file_content == content_j:
                row.append(0)
            else:
                row.append(len(compressor.compress((file_content+content_j).encode('latin-1'))))

        return row

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
        print('Compressing Files...')
        with tqdm(total = len(files_content['content'])) as pbar:
            row = []
            for content in files_content['content']:
                row.append(self.compress_files([content,compressors[compressor]]))
                pbar.update(1)

            compressed_matrix.append(row)

#         compressed_matrix.append([len(compressors[compressor].compress((content).encode('latin-1'))) for content in files_content['content']])

        # Compress appended files (p.e K(file1+file2))
        print('Compressing Appended Files...')
        with tqdm(total = len(files_content['content'])) as pbar:
            for file_content in files_content['content']:
                compressed_matrix.append(self.compress_appended_files([file_content, files_content['content'], compressors[compressor]]))
                pbar.update(1)

        return compressed_matrix

    ###### NORMAL COMPRESISON PARALEL######
    def normal_compression_paralel(self,compressor,compressors,n_processes):

        """
        # Performs normal copression i.e compresses all the file content together, without taking into account a possible columnar structure with paralel processes.
        # Outputs a matrix with 1st row having the files compressed sizes, and the other rows having the appended files compressed size.
        """

        # Get files content
        files_content = self.get_files_content()

        # Initialize matrix
        compressed_matrix = []

        # Paralel file compression
        params = [(file_content, compressors[compressor]) for file_content in files_content['content']]
        print('Compressing Files...')
        with Pool(n_processes) as executor:
            results = list( 
                tqdm(
                    executor.imap(self.compress_files, params),
                    total=len(params)
                )
            )
        compressed_matrix.append(results)

        # Paralel appended file compression
        params = [(file_content, files_content['content'], compressors[compressor]) for file_content in files_content['content']]
        print('Compressing Appended Files...')
        with Pool(n_processes) as executor:
            results = list( 
                tqdm(
                    executor.imap(self.compress_appended_files, params),
                    total=len(params)
                )
            )
            compressed_matrix = compressed_matrix + results

        return compressed_matrix

    ###### COMPRESS FILES BY COLUMN ######
    def compress_file_by_column(self,inp):

        file_content = inp[0]
        compressor = inp[1]
        compressors = inp[2]
        weights = inp[3]

        file_length = 0
        for col_i,col in enumerate(file_content):

            # If compressor is a sigle string, use it to compress all the columns.
            if  isinstance(compressor,str):
                file_length += (len(compressors[compressor].compress(np.ascontiguousarray(col))) * weights[col_i])

            # If compressor is a list of size equal to the number of columns, use same index compressors to compress same index columns.
            elif  isinstance(compressor[col_i],str):
                file_length += (len(compressors[compressor[col_i]].compress(np.ascontiguousarray(col)))* weights[col_i])

        return file_length

    ###### COMPRESS FILES BY COLUMN ######
    def compress_appended_files_by_column(self,inp):

        file_content = inp[0]
        file_index = inp[1]
        files_content = inp[2]
        compressor = inp[3]
        compressors = inp[4]
        weights = inp[5]

        row = []
        for j,content_j in enumerate(files_content):
            if file_index == j:
                row.append(0)
            else:

                # Load files with only one row.
                # A different way was implemented sice np.hstack did not behave the way we wanted when the files had a single row.
                if file_content.ndim == 1:
                    content_aux = np.core.defchararray.add(file_content, content_j)
                    content = [ [k] for k in content_aux]

                # Load files with more than a sigle row.
                # np.hstack, appends the files columns together.
                else:
                    content = np.hstack((file_content,content_j))

                # Compress appended columns
                file_length = 0
                for col_i,col in enumerate(content):

                    # If compressor is a sigle string, use it to compress all the columns.
                    if  isinstance(compressor,str):
                        file_length += (len(compressors[compressor].compress(np.ascontiguousarray(col)))* weights[col_i])

                    # If compressor is a list of size equal to the number of columns, use same index compressors to compress same index columns.
                    elif  isinstance(compressor[col_i],str):
                        file_length += (len(compressors[compressor[col_i]].compress(np.ascontiguousarray(col)))* weights[col_i])

                row.append(file_length)

        return row

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
        print('Compressing Files...')
        with tqdm(total = len(files_content['content'])) as pbar:
            row = []
            for file_content in files_content['content']:
                row.append(self.compress_file_by_column([file_content,compressor,compressors,weights]))
                pbar.update(1)

        # Append first row containing compressed file sizes to matrix
        compressed_matrix.append(row)

        print('Compressing appended Files...')
        # Compress appended files (p.e K(file1+file2))

        with tqdm(total = len(files_content['content'])**2) as pbar:
            for file_index,file_content in enumerate(files_content['content']):
                compressed_matrix.append(self.compress_appended_files_by_column([file_content,file_index,files_content['content'],compressor,compressors,weights]))
                pbar.update(1)

        print(compressed_matrix)
        return compressed_matrix

    ###### COMPRESSION BY COLUMN ######
    def compression_by_column_paralel(self, compressor, compressors, delimiter, weights,n_processes):

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
        print('Compressing Files...')
        params = [(file_content, compressor, compressors, weights) for file_content in files_content['content']]
        with Pool(n_processes) as executor:
            results = list( 
                tqdm(
                    executor.imap(self.compress_file_by_column, params),
                    total=len(params)
                )
            )

        compressed_matrix.append(results)

        # Compress appended files (p.e K(file1+file2))
        print('Compressing appended Files...')
        params = [(file_content, file_index, files_content['content'], compressor, compressors, weights) for file_index, file_content in enumerate(files_content['content'])]
        with Pool(n_processes) as executor:
            results = list( 
                tqdm(
                    executor.imap(self.compress_appended_files_by_column, params),
                    total=len(params)
                )
            )
            compressed_matrix = compressed_matrix + results

        return compressed_matrix


    ###### COMPUTE PAIR DISTANCES ######

    # GET COMPRESSED SIZES
    def get_compressed_sizes(self,compressor, compress_by_col, delimiter, weights, n_processes):

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
        if compress_by_col == 0 and n_processes == None:
            compressed_matrix = self.normal_compression(compressor,compressors)

        # By column: one compressor for all columns
        elif compress_by_col == 1 and n_processes == None:
            compressed_matrix = self.compression_by_column(compressor, compressors, delimiter, weights)

        # By column: one compressor for all columns
        elif compress_by_col == 0 and n_processes != None:
            compressed_matrix = self.normal_compression_paralel(compressor,compressors,n_processes)

        # By column: one compressor for all columns
        elif compress_by_col == 1 and n_processes != None:
            compressed_matrix = self.compression_by_column_paralel(compressor, compressors, delimiter, weights,n_processes)

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

    # COMPUTE UNCOMPRESSED MATRIX
    def get_ncd_quarter(self,inp):

        file_sizes = inp[0]
        append_file_sizes = inp[1]
        sli = inp[2]

        data = ''
        distance_matrix =  []

        for i, k1 in enumerate(file_sizes[sli[0]:sli[1]]):
            index = i + sli[0]
            row = []
            data += self.file_names[index]
            for j, k2 in enumerate(file_sizes):

                # Do not compute distance between the same file
                if index == j:
                    dist = 0.0

                else:
                    # COMPUTE DESIRED DISTANCE
                    dist = (append_file_sizes[index][j] - min([k1,k2])) / max([k1,k2])

                # Append to distmatrix.txt data string
                data = data + ' ' + str(round(dist,6))
                row.append(round(dist,6))        
            data += '\n'
            distance_matrix.append(row)

        return([data,distance_matrix])


    # COMPUTE NCD
    def distance_matrix(self, compressor, dm_name='distmatrix', output_path = None, compress_by_col = 0, delimiter = ',', weights = None, n_processes = None, compressed_matrix = None, verbose = 1):

        """
        Computes the normalized compression distance between all documents inside the folder.

        Parameters
        ----------
        compressor : {'zlib','gzip','bzlib','lzma','raw'},
            Wich compressor to use. Diferent compressors may yield diferent results when clustering.

            - 'zlib' produces asymmetrical matrices and smaller sizes for small strings.
            - 'gzip'  has simillar behavior to zlib (normally with bigger compressed sizes).
            - 'bzlib' produces symmetrical matrices and is recomended when using data that was encoded using the zgli.encode_df function.
            - 'lzma' usually produces the most compressed sizes.
            - 'raw' uses the raw file sizes i.e files do not get compressed.

        output_path : string, default=None 
            The file path of the to where the distance matrix should be outputed.

        compress_by_col : bool, default=False 
            Defines if the data shoul be compressed by column or normally. Default = False, i.e the files are compressed normally.

        delimiter : string, default=',' 
            The character to be used to separete columns, default = ',' since .csv is a common format for tabular data.

        weights : list, default=None
            A product between weight[i] and column[i] is computed if weights are provided. Default is none i.e all columns have weight = 1.

        verbose : int, default=False 
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

        if compressed_matrix  == None:
            # Get compressed size matrix
            uncompressed_matrix = self.get_uncompressed_sizes()
            compressed_matrix = self.get_compressed_sizes(compressor, compress_by_col, delimiter, weights, n_processes)

            # Get file sizes and appended file sizes to compute the NCD
            file_sizes = compressed_matrix[0]
            append_file_sizes = compressed_matrix[1:]

        else:
            # Get file sizes and appended file sizes to compute the NCD
            file_sizes = compressed_matrix[0]
            append_file_sizes = compressed_matrix[1]


        # Initialize data string, to be used as file content if necessary
        data = ''
        distance_matrix =  []

        if n_processes == None:
            # Normal matrix construction
            with tqdm(total = len(file_sizes)) as pbar:
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
                    pbar.update(1)

        elif n_processes != None:
            print('Computing distances')

            # Get ranges when dividing matrix by 4
            slices = []
            last_index = 0
            ru = int(math.ceil(len(file_sizes)/4))
            rd = int(math.trunc(len(file_sizes)/4))

            for i in range(0,4):
                if i < 3:
                    slices.append((last_index,last_index + ru))

                else:
                    slices.append((last_index,last_index + rd))

                last_index += ru

            # Paralel matrix construction
            quarters = np.array_split(file_sizes,4)
            params = [[file_sizes, append_file_sizes, sli] for sli in slices]
            with Pool(n_processes) as executor:
                results = list( 
                    tqdm(
                        executor.imap(self.get_ncd_quarter, params),
                        total=len(params)
                    )
                )

            # Construct matrix


            # Construct file data
            data = ''
            for r in results:
                data += r[0]

#             print(results[0][0])


        # Crete distance matrix file
        if output_path != None:
            dm_name = dm_name + '.txt'
            f = open(os.path.join(os.path.join(output_path,dm_name)), "w")
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