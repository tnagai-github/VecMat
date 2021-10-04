//  This program is written by Tetsuro Nagai

// Simple vecotor and matrix calculation
// Broad cast is impossible, unlike numpy.
// Operator "*" is used as the inner product, NOT as the product of each element.
// Simple and fundermental methods for matrix are implemented.
// For more complex methods, lapack should be considered. 
// Instead of the naive implementaiton, corresponding function from cblas 
// can be called by defining "vecNd_BLAS". 
// Athgough not tested, this should be be important when the dimension of problems is large. 

//include guard
#ifndef vecNd_FILE_
#define vecNd_FILE_

#include<iostream>
#include <initializer_list>
#include "prettyprint.hpp"

#ifdef DEBUG
#define bDEBUG (true)
#else
#define bDEBUG (false)
#endif

#define BOUNDCHECK (false)
//#define BOUNDCHECK (true)
//#ifdef BOUNDCHECK
//#define BOUNDCHECK (true)
//#else
//#define BOUNDCHECK (false)
//#endif

#ifdef vecNd_BLAS
#include <lapacke.h>
#include <cblas.h>
#endif

/*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*/
namespace VecMat {
    template <int VDIM > 
    class vecNd {
        public: 
        double vec[VDIM];
        double * const p = vec;
        static constexpr int size = VDIM;
        
        //double*  begin(){return vec;};
        //double*  end()  {return vec+VDIM;};

        //copy construstor
        vecNd(const vecNd & ) = default;
        vecNd& operator= (const vecNd &) = default;

        //move constructor
        vecNd(vecNd && ) = default;
        vecNd& operator= (vecNd &&)  = default;

        vecNd(std::initializer_list<double> list){
            if(list.size() != size){
                std::cerr << "warning: length of initializer list is not consistent!! " << std::endl;
            }
            auto itr = list.begin();
            for (int i = 0; i<size; ++i){
                vec[i]=*itr;
                itr++;
            }
        }

        vecNd(){
            for(auto& each : vec){
                each=0.0;
            }
        }
        vecNd(const double x){
            for(int i = 0; i< size; ++i ){
                vec[i]=x;
            }
        }

        vecNd(const double vec_in[VDIM]){
            for(int i = 0; i< size; ++i ){
                vec[i]=vec_in[i];
            }
        }

        inline double dot (const vecNd &a) const ;

        double abs () const {
            return sqrt(this->dot(*this));
        }

        double sum () const {
            double sum = 0.0;
            for(int i = 0; i<VDIM; ++i){
                sum+=vec[i];
            }
            return sum;
        }

        inline double & operator [] (const int i );
        inline double   operator [] (const int i ) const;

        vecNd& operator+= (const vecNd & a) {
            for(int i=0; i<size; i++){
                this->vec[i]+=a.vec[i];
            }
            return *this;
        } 

        vecNd& operator-= (const vecNd & a) {
            for(int i=0; i<size; i++){
                this->vec[i]-=a.vec[i];
            }
            return *this;
        } 

        vecNd& operator*= (const double a) {
            #ifdef vecNd_BLAS 
            cblas_dscal(this->size, a, this->vec, 1);
            return *this;
            //cblas_dscal(const int N, const double alpha, double *X, const int incX);
            #endif
            for(int i=0; i<size; i++){
                this->vec[i]*=a;
            }
            return *this;
        } 

        vecNd& operator/= (const double a) {
            double tmp = 1.0/a;
            return (*this)*=tmp;
        } 
    };

    template <int VDIM > 
    double vecNd<VDIM>::dot (const vecNd &a) const {
        #ifdef vecNd_BLAS
        return cblas_ddot(this->size, this->vec, 1, a.vec,1);
        #else
        double sum=0;
        for(int i=0; i < VDIM; ++i){           
            sum+=a.vec[i]*vec[i];
        }
        return sum;
        #endif
    }

    template <int VDIM > 
    double & vecNd<VDIM>::operator [] (const int i ) {
        if(BOUNDCHECK){
            if(i<0 or VDIM<=i){
                std::cerr << "Improper access" << std::endl;
                exit(EXIT_FAILURE) ;
            }
        }
        return this->vec[i];
    }

    template <int VDIM > 
    double vecNd<VDIM>::operator [] (const int i ) const {
        if(BOUNDCHECK){
            if(i<0 or VDIM<=i){
                std::cerr << "Improper access" << std::endl;
                exit(EXIT_FAILURE) ;
            }
        }
        return this->vec[i];
    }

    template <int VDIM > 
    double dot(const vecNd<VDIM> &a, const vecNd<VDIM> &b){
        return a.dot(b);
    }

    template <int VDIM > 
    double abs(const vecNd<VDIM> &a){
        return sqrt(dot(a, a));
    }

    template <int VDIM > 
    std::ostream& operator << (std::ostream &os, const vecNd<VDIM>& vv){
        os << vv.vec ; //with prittyprint 
        return os;
    }


    template <int VDIM > 
    const vecNd<VDIM> operator + (const vecNd<VDIM> &a, const vecNd<VDIM> &b){
        return vecNd<VDIM>(a)+=b;
    }

    template <int VDIM > 
    const vecNd<VDIM> operator - (const vecNd<VDIM> &a, const vecNd<VDIM> &b){
        return vecNd<VDIM>(a)-=b;
    }
    template <int VDIM > 
    const vecNd<VDIM> operator * (const double a, const vecNd<VDIM>&b){
        return vecNd<VDIM>(b)*=a;
    }

    template <int VDIM > 
    double operator * (const vecNd<VDIM> &a, const vecNd<VDIM> &b){
        return dot(a,b);
    }

    template <int VDIM > 
    const vecNd<VDIM> operator * (const vecNd<VDIM> &a, const double b){
        return vecNd<VDIM>(a)*=b;
    }

    template <int VDIM > 
    const vecNd<VDIM> operator/ (const vecNd<VDIM> &a, const double b){
        return vecNd<VDIM>(a)/=b;
    }


    template <int VDIM > 
    class matNd {
        public:
        double mat[VDIM][VDIM] ;
        double * const p = mat[0];
        static constexpr int size = VDIM;
            
        //move constructor
        matNd(matNd&&) = default;
        matNd& operator=(matNd&&) = default;

        matNd(std::initializer_list<double> list){
            if(list.size() != size*size){
                std::cerr << "warning: length of initializer list is not consistent!! " << std::endl;
            }
            auto itr = list.begin();
            for(int i = 0; i<VDIM; ++i){
                for(int j = 0; j<VDIM; j++){
                    mat[i][j] = *itr;
                    itr++;
                }
            }
        }

        matNd(const matNd<VDIM> &obj){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    this->mat[i][j] = obj.mat[i][j];
                }
            }
        }

        matNd& operator=(const matNd & a){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j] = a[i][j] ;
                }
            }
            return *this;
        }

        matNd(){
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

        matNd(double x){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j] = x ;
                }
            }
        }

        matNd(double mat_in[VDIM][VDIM]){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j] = mat_in[i][j] ;
                }
            }
        }

        matNd(vecNd<VDIM> vec[VDIM]){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j] = vec[i][j] ;
                }
            }
        }

        double* operator [] (const int i ){
            if(BOUNDCHECK){
                if(i<0 or VDIM<=i){
                    std::cerr << "Improper access" << std::endl;
                    exit(EXIT_FAILURE) ;
                }
            }
            return this->mat[i];
        }

        const double* operator [] (const int i ) const {
            if(BOUNDCHECK){
                if(i<0 or VDIM<=i){
                    std::cerr << "Improper access" << std::endl;
                    exit(EXIT_FAILURE) ;
                }
            }
            return this->mat[i];
        }

        const double* row(int i) const {
            return (*this)[i];
        }
        double* row(int i) {
            return (*this)[i];
        }

        matNd& operator+=(const matNd & a){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j] += a[i][j] ;
                }
            }
            return *this;
        }

        matNd& operator-=(const matNd & a){
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j] -= a[i][j] ;
                }
            }
            return *this;
        }

        matNd& operator*= (const double a) {
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j]*=a ;
                }
            }
            return *this;
        } 

        matNd& operator*= (const matNd a) {
            #ifdef vecNd_BLAS
            matNd<VDIM> results(0.0);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,
                 CblasNoTrans, VDIM, VDIM,
                 VDIM, 1.0, this->p,
                 VDIM, a.p, VDIM,
                  0.0, results.p, VDIM);
            #else
            matNd results(0.0); 
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

        matNd& operator/= (const double a) {
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    mat[i][j]/=a ;
                }
            }
            return *this;
        } 

        const matNd T() const{
            matNd result;
            for(int i =0 ; i < VDIM; ++i){                    
                for(int j =0 ; j < VDIM; ++j){                    
                    result[i][j] = (*this)[j][i] ;
                }
            }
            return result;
        }
        //const matNd pivotA() const ;
        const matNd triangle() const ;
        double det() const ;


    };
        
    template<int VDIM>
    class t_ans_ev{
        public:
        double eigenValRe[VDIM] ={};
        double eigenValIm[VDIM] ={};
        matNd<VDIM> eigenVecl=0.0;
        matNd<VDIM> eigenVecr=0.0;
        vecNd<VDIM> eigenVeclv[VDIM];
        vecNd<VDIM> eigenVecrv[VDIM];
    };

    template<int VDIM>
    std::ostream& operator << (std::ostream &os, const t_ans_ev<VDIM>& obj){
        for(int i=0; i<VDIM; i++){
            os << "Eigen val[ " << i << " ]= " << obj.eigenValRe[i] <<" + " << obj.eigenValIm[i] <<" *i" <<std::endl;//with pritty print 
        }
        os << "Eigen vec (left)" <<std::endl;
        for(int i=0; i<VDIM; i++){
            os  <<  vecNd<VDIM> (obj.eigenVecl[i]) << std::endl ;
        }
        os << "Eigen vec (right)" <<std::endl;
        for(int i=0; i<VDIM; i++){
            os  <<  vecNd<VDIM> (obj.eigenVecr[i]) ;
            if(i!=VDIM-1){
                os  <<  std::endl;
            }
        }
        return os;
    }

    template<int VDIM>
    t_ans_ev<VDIM> wrap_dgeev(const matNd<VDIM> &obj) {
        matNd<VDIM> copy = obj;
        t_ans_ev<VDIM>  ans;
        LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'V', 'V', 
           VDIM, copy.p, VDIM, ans.eigenValRe, 
           ans.eigenValIm, ans.eigenVecl.p, VDIM,  ans.eigenVecr.p, 
           VDIM);
        ans.eigenVecl=ans.eigenVecl.T();
        ans.eigenVecr=ans.eigenVecr.T();
        for(int i =0; i<VDIM; ++i){
        for(int j =0; j<VDIM; ++j){
            ans.eigenVeclv[i][j] = ans.eigenVecl[i][j];
            ans.eigenVecrv[i][j] = ans.eigenVecr[i][j];
        }
        }
        return ans;
    }

    template<int VDIM>
    std::ostream& operator << (std::ostream &os, const matNd<VDIM>& mat){
        os << mat.mat ; //with pritty print 
        return os;
    }

    template<int VDIM>
    const matNd<VDIM>  operator+ (const matNd<VDIM> &a, const matNd<VDIM> &b){
        return matNd<VDIM>(a)+=b;
    }

    template<int VDIM>
    const matNd<VDIM>  operator- (const matNd<VDIM> &a, const matNd<VDIM> &b){
        return matNd<VDIM>(a)-=b;
    }

    template<int VDIM>
    const matNd<VDIM>  operator* (const matNd<VDIM> &a, const matNd<VDIM> &b){
        return matNd<VDIM>(a)*=b;
    }

    template<int VDIM>
    const matNd<VDIM>  operator* (const matNd<VDIM> &a, const double b){
        return matNd<VDIM>(a)*=b;
    }

    template<int VDIM>
    const matNd<VDIM>  operator* (const double a, const matNd<VDIM> &b){
        return matNd<VDIM>(b)*=a;
    }

    template<int VDIM>
    const vecNd<VDIM>  operator* (const matNd<VDIM> &a, const vecNd<VDIM> &b){
        vecNd<VDIM> result(0.0);
        #ifdef vecNd_BLAS
        cblas_dgemv(CblasRowMajor, 
            CblasNoTrans, /*M=*/VDIM, /*N=*/ VDIM,
            /*alpha=*/1.0, a.p, /*lda=*/ VDIM,
            b.p, 1, /*beta=*/0.0, 
            result.p, 1);
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
    const vecNd<VDIM> operator* (const vecNd<VDIM> &a, const matNd<VDIM> &b){
        vecNd<VDIM> result(0.0);
        #ifdef vecNd_BLAS
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


    template<int VDIM>
    const matNd<VDIM> pow (const matNd<VDIM> &a, const int n){
        if(n == 0){
            matNd<VDIM> I;
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
    const matNd<VDIM>  operator/ (const matNd<VDIM> &a, const double b){
        return matNd<VDIM>(a)/=b;
    }


    // outerproduct is defined for N=VDIM only. this should be practical enough.
    
    const vecNd<3> cross(const vecNd<3> &a, const vecNd<3> &b){
        int VDIM=3;
        vecNd<3> result(0.0);
        for (int i=0 ; i <VDIM ; ++i){
            result[i]+=a[(i+1)%VDIM]*b[(i+2)%VDIM];
        }
        for (int i=0 ; i <VDIM ; ++i){
            result[i]-=a[(i+2)%VDIM]*b[(i+1)%VDIM];
        }
        return result;
    }


    template<int VDIM>
    const matNd<VDIM> matNd<VDIM>::triangle () const {
        matNd<VDIM> result=(*this);

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
    const matNd<VDIM> triangle (const matNd<VDIM> & a )  {
        return a.triangle();
    }

    template<int VDIM>
    double matNd<VDIM>::det () const {
        matNd<VDIM> tmp(this->triangle());
        double result=1.0;
        for(int i=0; i<VDIM; i++){
            result*=tmp[i][i];
        }
        return result;
    }

    template<int VDIM>
    double det (matNd<VDIM> const &a){
        return a.det();
    }

    // some more special function for VDIM = 3
    const matNd<3> rot_by_x(double theta){
        const int VDIM=3;
        double array[VDIM][VDIM]={{1, 0, 0}, {0, cos(theta), -sin(theta)}, {0, sin(theta), cos(theta)}};
        matNd<VDIM> result(array);
        return result;
    }

    const matNd<3> rot_by_y(double theta){
        const int VDIM=3;
        double array[VDIM][VDIM]={{cos(theta), 0, sin(theta)}, {0, 1, 0}, {-sin(theta), 0, cos(theta)}};
        matNd<VDIM> result(array);
        return result;
    }

    const matNd<3> rot_by_z(double theta){
        const int VDIM=3;
        double array[VDIM][VDIM]={{cos(theta), -sin(theta), 0}, {sin(theta), cos(theta), 0}, {0, 0, 1}};
        matNd<VDIM> result(array);
        return result;
    }
}


#endif //for include guard
