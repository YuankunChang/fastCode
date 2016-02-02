/*

    Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix_mul.h"
#include <pmmintrin.h>

namespace omp
{
  void
  matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
  {
      unsigned int i,j,k;
      __m128 X, Y;
      
      float sq_matrix_3[sq_dimension * sq_dimension];
      
      #pragma omp parallel for
      for(i=0; i<sq_dimension; i++){
          
          int row = i * sq_dimension;
          int row2 = 0;
          for(unsigned j=0; j<sq_dimension; j++){
              sq_matrix_3[row + j] = sq_matrix_2[row2 + i];
              row2 = row2 + sq_dimension;
          }
      }
      
      //method 1
//
//      
//      
//    #pragma omp parallel for
//      for (i = 0; i < sq_dimension; i++)
//      {
//          for(j = 0; j < sq_dimension; j++)
//          {
//              int row = i * sq_dimension;
//              int row2 = j * sq_dimension;
//              int num = row + j;
//              sq_matrix_result[num] = 0;
//              for (k = 0; k < sq_dimension; k++) {
//                  sq_matrix_result[num] = sq_matrix_result[num] + sq_matrix_1[row + k] * sq_matrix_3[row2 + k];
//              }
//          }
//      }// End of parallel region

      for (i = 0; i < sq_dimension; i++) {
          for (j = 0; j < sq_dimension; j++) {
              int row = i * sq_dimension;
              int row2 = j * sq_dimension;
              int num = row + j;
              sq_matrix_result[num] = 0;
              
              __m128 acc = _mm_setzero_ps();
              float inner_prod = 0, temp[4] = {0, 0, 0, 0};
              
              for (k = 0; k < sq_dimension - 4; k += 4) {
                  X = _mm_load_ps(sq_matrix_1 + row + k);
                  Y = _mm_load_ps(sq_matrix_3 + row2 + k);
                  acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
//                  sq_matrix_result[num] = sq_matrix_result[num] + sq_matrix_1[row + k] * sq_matrix_3[row2 + k];
              }
              _mm_store_ps(&temp[0], acc);
              inner_prod = temp[0] + temp[1] + temp[2] + temp[3];
              
              for(; k < sq_dimension; k++)
                  inner_prod += sq_matrix_1[row + k] * sq_matrix_3[row2 + k];
              
              sq_matrix_result[num] = inner_prod;

          }
      }
      
  }
  
} //namespace omp
