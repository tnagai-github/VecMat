//  This program is written by Tetsuro Nagai

// Simple vecotor and matrix calculation
// No broad cast, unlike numpy.
// * is used as inner product, NOT as the product of each element.
// very simple and fundermental method for matrix is here implemented.
// For more complex methods, lapack should be considered. 

//incude guard
#ifndef __vectorNd_FILE__
#define __vectorNd_FILE__

#include<iostream>
#include "prettyprint.hpp"

#ifdef DEBUG
#define bDEBUG (true)
#else
#define bDEBUG (false)
#endif

//#undef vectorNd_BLASLAPACK
//#define vectorNd_BLASLAPACK

#ifdef vectorNd_BLASLAPACK
#include <lapacke.h>
#include <cblas.h>
#endif

// +>+>+>+>+>+>+>+>+>+>+>+>+>+>+>+>+>+>+>+>+>+>+>>+>+>+>+>+>+>
namespace VecMat {
    template <int VDIM > 
    class vectorNd {
        public: 
        int size = VDIM;
        double vec[VDIM];

        vectorNd(){
            for(int i = 0; i< size; ++i ){
                vec[i]=0.0;
            }
        }
        vectorNd(const double x){
            for(int i = 0; i< size; ++i ){
                vec[i]=x;
            }
        }
        vectorNd(double vec_in[VDIM]){
            for(int i = 0; i< size; ++i ){
                vec[i]=vec_in[i];
            }
        }

        double dot (const vectorNd &a) const {
            #ifdef vectorNd_BLASLAPACK
            return cblas_ddot(this->size, this->vec, 1, a.vec,1);
            #else
            double sum=0;
            for(int i=0; i < VDIM; ++i){           
                sum+=a.vec[i]*vec[i];
            }
            return sum;
            #endif
        }

        double abs () const {
            //return sqrt(dot(*this, *this));
            return sqrt(this->dot(*this));
        }

        double & operator [] (const int i );
        double   operator [] (const int i ) const;

        vectorNd& operator+= (const vectorNd & a) {
            for(int i=0; i<size; i++){
                this->vec[i]+=a.vec[i];
            }
            return *this;
        } 

        vectorNd& operator-= (const vectorNd & a) {
            for(int i=0; i<size; i++){
                this->vec[i]-=a.vec[i];
            }
            return *this;
        } 

        vectorNd& operator*= (const double a) {
            #ifdef vectorNd_BLASLAPACK 
            cblas_dscal(this->size, a, this->vec, 1);
            return *this;
            //cblas_dscal(const int N, const double alpha, double *X, const int incX);
            #endif
            for(int i=0; i<size; i++){
                this->vec[i]*=a;
            }
            return *this;
        } 

        vectorNd& operator/= (const double a) {
            double tmp = 1.0/a;
            //for(int i=0; i<size; i++){
            //    this->vec[i]*=tmp;
            //}
            //return *this;
            return (*this)*=tmp;
        } 
    };

    template <int VDIM > 
    double dot(const vectorNd<VDIM> &a, const vectorNd<VDIM> &b){
        return a.dot(b);
    }

    template <int VDIM > 
    double abs(const vectorNd<VDIM> &a){
        return sqrt(dot(a, a));
    }


    template <int VDIM > 
    std::ostream& operator << (std::ostream &os, const vectorNd<VDIM>& vv){
        os << vv.vec ; //with prittyprint 
        return os;
    }

    template <int VDIM > 
    double & vectorNd<VDIM>::operator [] (const int i ) {
        if(bDEBUG){
            if(i<0 or VDIM<=i){
                std::cerr << "Improper access" << std::endl;
                exit(EXIT_FAILURE) ;
            }
        }
        return this->vec[i];
    }

    template <int VDIM > 
    double vectorNd<VDIM>::operator [] (const int i ) const {
        if(bDEBUG){
            if(i<0 or VDIM<=i){
                std::cerr << "Improper access" << std::endl;
                exit(EXIT_FAILURE) ;
            }
        }
        return this->vec[i];
    }

    template <int VDIM > 
    const vectorNd<VDIM> operator + (const vectorNd<VDIM> &a, const vectorNd<VDIM> &b){
        return vectorNd<VDIM>(a)+=b;
    }

    template <int VDIM > 
    const vectorNd<VDIM> operator - (const vectorNd<VDIM> &a, const vectorNd<VDIM> &b){
        return vectorNd<VDIM>(a)-=b;
    }
    template <int VDIM > 
    const vectorNd<VDIM> operator * (const double a, const vectorNd<VDIM>&b){
        return vectorNd<VDIM>(b)*=a;
    }

    template <int VDIM > 
    double operator * (const vectorNd<VDIM> &a, const vectorNd<VDIM> &b){
        return dot(a,b);
    }

    template <int VDIM > 
    const vectorNd<VDIM> operator * (const vectorNd<VDIM> &a, const double b){
        return vectorNd<VDIM>(a)*=b;
    }

    template <int VDIM > 
    const vectorNd<VDIM> operator/ (const vectorNd<VDIM> &a, const double b){
        return vectorNd<VDIM>(a)/=b;
    }


    template <int VDIM > 
    class matrix {            
        public:
        double mat[VDIM][VDIM] ;

        matrix(){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    if(i==j){
                        mat[i][j]=1.0; 
                    }else{
                        mat[i][j]=0.0;
                    }
                }
            }
        }
        matrix(double x){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j] = x ;
                }
            }
        }

        matrix(double mat_in[VDIM][VDIM]){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j] = mat_in[i][j] ;
                }
            }
        }

        double* operator [] (const int i ){
            if(bDEBUG){
                if(i<0 or VDIM<=i){
                    std::cerr << "Improper access" << std::endl;
                    exit(EXIT_FAILURE) ;
                }
            }
            return this->mat[i];
        }

        const double* operator [] (const int i ) const {
            if(bDEBUG){
                if(i<0 or VDIM<=i){
                    std::cerr << "Improper access" << std::endl;
                    exit(EXIT_FAILURE) ;
                }
            }
            return this->mat[i];
        }

        matrix& operator+=(const matrix & a){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j] += a[i][j] ;
                }
            }
            return *this;
        }

        matrix& operator-=(const matrix & a){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j] -= a[i][j] ;
                }
            }
            return *this;
        }

        matrix& operator*= (const double a) {
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j]*=a ;
                }
            }
            return *this;
        } 

        matrix& operator*= (const matrix a) {
            #ifdef vectorNd_BLASLAPACK
            matrix<VDIM> results(0.0);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,
                 CblasNoTrans, VDIM, VDIM,
                 VDIM, 1.0, this->mat[0],
                 VDIM, a.mat[0], VDIM,
                  0.0, results.mat[0], VDIM);
            //void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
            //     CBLAS_TRANSPOSE TransB, const int M, const int N,
            //     const int K, const double alpha, const double *A,
            //     const int lda, const double *B, const int ldb,
            //     const double beta, double *C, const int ldc);
            #else
            matrix results(0.0); 
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                   for(int k =0 ; k < VDIM; ++k){                    
                        results.mat[i][j]+=mat[i][k]*a[k][j];
                   }
                }
            }
            #endif
            *this=results;
            return *this;

        } 

        matrix& operator/= (const double a) {
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j]/=a ;
                }
            }
            return *this;
        } 

        const matrix T() const{
            matrix result;
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    result[i][j] = (*this)[j][i] ;
                }
            }
            return result;
        }
        //const matrix pivotA() const ;
        const matrix triangle() const ;
        double det() const ;
    };

    template<int VDIM>
    std::ostream& operator << (std::ostream &os, const matrix<VDIM>& mat){
        os << mat.mat ; //with pritty print 
        return os;
    }

    template<int VDIM>
    const matrix<VDIM>  operator+ (const matrix<VDIM> &a, const matrix<VDIM> &b){
        return matrix<VDIM>(a)+=b;
    }

    template<int VDIM>
    const matrix<VDIM>  operator- (const matrix<VDIM> &a, const matrix<VDIM> &b){
        return matrix<VDIM>(a)-=b;
    }

    template<int VDIM>
    const matrix<VDIM>  operator* (const matrix<VDIM> &a, const matrix<VDIM> &b){
        return matrix<VDIM>(a)*=b;
    }

    template<int VDIM>
    const matrix<VDIM>  operator* (const matrix<VDIM> &a, const double b){
        return matrix<VDIM>(a)*=b;
    }

    template<int VDIM>
    const matrix<VDIM>  operator* (const double a, const matrix<VDIM> &b){
        return matrix<VDIM>(b)*=a;
    }

    template<int VDIM>
    const vectorNd<VDIM>  operator* (const matrix<VDIM> &a, const vectorNd<VDIM> &b){
        vectorNd<VDIM> result(0.0);
        #ifdef vectorNd_BLASLAPACK
        cblas_dgemv(CblasRowMajor, 
            CblasNoTrans, VDIM, VDIM,
            1.0, a.mat[0], VDIM,
            b.vec, 1, 0.0, 
            result.vec, 1);
        //void cblas_dgemv(CBLAS_LAYOUT layout,
        //         CBLAS_TRANSPOSE TransA, const int M, const int N,
        //         const double alpha, const double *A, const int lda,
        //         const double *X, const int incX, const double beta,
        //         double *Y, const int incY);
        #else
        for(int i=0; i < VDIM; i++){
            for(int j=0; j < VDIM; j++){
                result[i]+=a[i][j]*b[j];
            }
        }
        #endif
        return result;
    }

    template<int VDIM>
    const vectorNd<VDIM> operator* (const vectorNd<VDIM> &a, const matrix<VDIM> &b){
        vectorNd<VDIM> result(0.0);
        #ifdef vectorNd_BLASLAPACK
        cblas_dgemv(CblasRowMajor, 
            CblasTrans, VDIM, VDIM,
            1.0, b.mat[0], VDIM,
            a.vec, 1, 0.0, 
            result.vec, 1);
        //void cblas_dgemv(CBLAS_LAYOUT layout,
        //         CBLAS_TRANSPOSE TransA, const int M, const int N,
        //         const double alpha, const double *A, const int lda,
        //         const double *X, const int incX, const double beta,
        //         double *Y, const int incY);
        #else
        for(int i=0; i < VDIM; i++){
            for(int j=0; j < VDIM; j++){
                result[i]+=a[j]*b[j][i];
            }
        }
        #endif
        return result;
    }

//   template<int VDIM>
//   const matrix<VDIM> pow (const matrix<VDIM> &a, const int n){
//       if(n == 0){
//           matrix<VDIM> I;
//           return I;
//       }
//       if(n < 0){
//           std::cerr <<"invalid power index. Index must be non-negative integer" << std::endl;
//           exit(EXIT_FAILURE);
//       }
//       //for n>1
//       return  pow(a, n-1) * a;
//   }

    template<int VDIM>
    const matrix<VDIM> pow (const matrix<VDIM> &a, const int n){
        if(n == 0){
            matrix<VDIM> I;
            return I;
        }
        if(n < 0){
            std::cerr <<"invalid power index. Index must be non-negative integer" << std::endl;
            exit(EXIT_FAILURE);
        }
        //for n>1
        if(n%2 == 0){
            return  pow(a*a, n/2) ;
        }
        else /*if(n%2 ==1)*/{
            return  a*pow(a*a, (n)/2) ;
        }
    }

    template<int VDIM>
    const matrix<VDIM>  operator/ (const matrix<VDIM> &a, const double b){
        return matrix<VDIM>(a)/=b;
    }


    // outerproduct is defined for N=VDIM only. this should be practical enough.
    
    const vectorNd<3> cross(const vectorNd<3> &a, const vectorNd<3> &b){
        int VDIM=3;
        vectorNd<3> result(0.0);
        for (int i=0 ; i <VDIM ; ++i){
            result[i]+=a[(i+1)%VDIM]*b[(i+2)%VDIM];
        }
        for (int i=0 ; i <VDIM ; ++i){
            result[i]-=a[(i+2)%VDIM]*b[(i+1)%VDIM];
        }
        return result;
    }


    template<int VDIM>
    const matrix<VDIM> matrix<VDIM>::triangle () const {
        matrix<VDIM> result=(*this);

        for(int i = 0; i <VDIM;  ++i){
            if (result[i][i] == 0.0){
                int j ;
                for(j =i+1; j<VDIM; ++j){
                    if(result[j][i] != 0.0 ){
                        break;
                    }
                }
                if(j!=VDIM){
                    for(int k=0;k<VDIM;k++){
                        result[i][k]+=result[j][k];
                    }
                }else{
                    if(bDEBUG){
                        std::cerr << "always zero for this column" << std::endl;
                        std::cerr << "thus, try pivot of columns" << std::endl;
                    }
                    int jj;
                    for (jj=i+1; jj<VDIM; jj++){
                        if(result[i][jj] != 0){
                            break;
                        }
                        if(jj!=VDIM){
                            for(int k=0; k<VDIM; k++){
                                result[k][i]+=result[k][jj];
                            }
                        }else{
                            if(bDEBUG){
                                std::cerr << "always zero for both row and column" << std::endl;
                                std::cerr << "not (yet) supported" << std::endl;
                                exit(EXIT_FAILURE);
                            }
                        }
                    }
                }
            }
            for(int j=i+1; j<VDIM; j++){
                //if(i==j){
                //    continue;
                //}
                if(result[i][i]!=0.0){
                double tmp = result[j][i]/result[i][i];
                    for (int k =0; k <VDIM; k++){
                        result[j][k]-=tmp*result[i][k];
                    }
                }
            }
        }
        return result;
    }

    template<int VDIM>
    const matrix<VDIM> triangle (const matrix<VDIM> & a )  {
        return a.triangle();
    }

    template<int VDIM>
    double matrix<VDIM>::det () const {
        matrix<VDIM> tmp(this->triangle());
        double result=1.0;
        for(int i=0; i<VDIM; i++){
            result*=tmp[i][i];
        }
        return result;
    }
    template<int VDIM>
    double det (matrix<VDIM> const &a){
        return a.det();
    }

    // some more special function for VDIM = 3
    const matrix<3> rot_by_x(double theta){
        const int VDIM=3;
        double array[VDIM][VDIM]={{1, 0, 0}, {0, cos(theta), -sin(theta)}, {0, sin(theta), cos(theta)}};
        matrix<VDIM> result(array);
        return result;
    }

    const matrix<3> rot_by_y(double theta){
        const int VDIM=3;
        double array[VDIM][VDIM]={{cos(theta), 0, sin(theta)}, {0, 1, 0}, {-sin(theta), 0, cos(theta)}};
        matrix<VDIM> result(array);
        return result;
    }

    const matrix<3> rot_by_z(double theta){
        const int VDIM=3;
        double array[VDIM][VDIM]={{cos(theta), -sin(theta), 0}, {sin(theta), cos(theta), 0}, {0, 0, 1}};
        matrix<VDIM> result(array);
        return result;
    }
}

#ifdef vectorNd_BLASLAPACK
#undef vectorNd_BLASLAPACK
#endif

#endif
