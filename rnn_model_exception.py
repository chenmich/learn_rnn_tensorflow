# Copyright 2017 The Chenmich Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''This is exception class for my model
'''
import sys
class RNNModelException(Exception):
    ''' RNN model's root Exception
    '''
class RNNModelDataNotCompatibleException(RNNModelException):
    pass
class RNNModelRawDataNotComptibleException(RNNModelDataNotCompatibleException):
    ''' This class define the exception for raw data
    '''
    def __init__(self, message, raw_file, row=0, column=0, *args):
        ''' args:
            rwa_file:the name of file which caused exception
            row:     line in the file  which caused exception
            column:  column in the file which caused exception
        '''
        self.__raw_file__ = raw_file
        self.__row__ = row
        self.__column__ = column
        self.__message__ = message
        super(RNNModelRawDataNotComptibleException, self).__init__(
            message, raw_file, row, column, *args)
class RNNModelDataLengthNotEnoughException(RNNModelDataNotCompatibleException):
    def __init__(self, message, raw_file, *args):
        self.__raw_file__ = raw_file
        self.__message__ = message
        self.__row__ = None
        self.__column__ = None
        super(RNNModelDataLengthNotEnoughException, self).__init__(raw_file, message, *args)
class RNNModelDataContentHasNonNumberException(RNNModelDataNotCompatibleException):
    pass

try:
    raise RNNModelDataLengthNotEnoughException('a', 'some.csv')
except RNNModelDataLengthNotEnoughException:
    sys_info = sys.exc_info()
    print(sys_info[1].__row__)

try:
    raise RNNModelRawDataNotComptibleException('a', 'some.csv', row=1, column=2)
except:
    sys_info = sys.exc_info()
    print(sys_info[1].__row__)