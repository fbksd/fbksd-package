
/*
    Copyright(c) 2012-2013 Tzu-Mao Li
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


#if defined(_MSC_VER)
#pragma once
#endif

#ifndef SBF_TWO_D_ARRAY__
#define SBF_TWO_D_ARRAY__

#include <algorithm>

/**
 *   @brief A class encapsulates a generic row-major 2d array.
 */
template<typename T>
class TwoDArray {
public:
    inline TwoDArray(int col=0, int row=0) {_init(col,row);}
    inline TwoDArray(int col, int row, T v) {_init(col,row); *this = v;}
    inline TwoDArray(const TwoDArray& ary) {
        _init(ary.m_Col, ary.m_Row);
        std::copy(ary.m_Data, ary.m_Data+(ary.m_Col*ary.m_Row), m_Data);
    }
    inline TwoDArray(int col,int row, T* vAry) {
        _init(col,row);
        std::copy(vAry, vAry+(col*row), m_Data);
    }
    inline virtual ~TwoDArray() {
        delete[] m_Data;
    }

    inline int GetColNum() const {return m_Col;}
    inline int GetRowNum() const {return m_Row;}

    TwoDArray& operator=(const TwoDArray &ary) {
        if(ary.m_Row != m_Row || ary.m_Col != m_Col) {
            delete[] m_Data;
            m_Row = ary.m_Row; m_Col = ary.m_Col;
            m_Data = new T[m_Row*m_Col];
        }
        std::copy(ary.m_Data, ary.m_Data+(m_Col*m_Row), m_Data);
        return *this;
    }

    TwoDArray& operator=(const T &v) {
        std::fill(m_Data,m_Data+(m_Col*m_Row),v);
        return *this;
    }
    inline T* operator[](int i) {return (m_Data+(m_Col*i));}
    inline T const * const operator[](int i) const {return (m_Data+(m_Col*i));}
    inline T& operator()(int i) {return m_Data[i];}
    inline const T& operator()(int i) const {return m_Data[i];}
    inline T& operator()(int c, int r) { return m_Data[r*m_Col+c];}
    inline const T& operator()(int c, int r) const {  return m_Data[r*m_Col+c];}

    T* GetRawPtr() {return m_Data;}
    const T* GetRawPtr() const {return m_Data;}
private:

    void _init(int col, int row) {
        m_Col = col;
        m_Row = row;
        m_Data = new T[m_Col*m_Row];
    }

    int m_Col;
    int m_Row;
    T* m_Data;
};

#endif //#ifndef SBF_TWO_D_ARRAY__
