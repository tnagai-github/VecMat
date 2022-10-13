//  This program is written by Tetsuro Nagai

// Simple vecotor and matrix calculation
// Broad cast is impossible, unlike numpy.
// Operator "*" is used as the inner product, NOT as the product of each element.
// Simple and fundermental methods for matrix are implemented.
// For more complex methods, lapack should be considered. 
// Instead of the naive implementaiton, cblas can be internally called by defining "vecNd_BLAS". 
// Athgough speeds have not been compared yet, blas should be advantageous when the dimensions of problem are large. 

//include guard
#ifndef vec_FILE_
#define vec_FILE_

#include<iostream>
#include <initializer_list>
#include "cxx-prettyprint/prettyprint.hpp"

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

#define vecNd_BLAS
#ifdef vecNd_BLAS
#include <lapacke.h>
#include <cblas.h>
#endif

#include <vector> 

/*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*/
namespace VecMat {
    class Vec {
        public: 
        std::vector<double> v;
        
        //vec()= delete; // zero vector
        Vec(const int dim, const double x=0);
        //vec(std::initializer_list<double> list);
        //Vec(const double vec_in[VDIM]);

        ////copy construstor
        Vec (const Vec & a) ;
        Vec (const std::vector<double> & a) ;
        Vec& operator= (const Vec & a);

        int size() const {return this->v.size();}

        double dot (const Vec &a) const ;
        double abs () const ;
        double sum () const ;

        double & operator [] (const int i );
        double   operator [] (const int i ) const;

        //double & at (const int i );
        //double   at (const int i ) const;

        Vec& operator+= (const Vec & a) ;
        Vec& operator-= (const Vec & a) ;

        Vec& operator*= (const double a) ;
        Vec& operator/= (const double a) ;

        const Vec operator+ ()const;
        const Vec operator- ()const;

    };
    double dot(const Vec &a, const Vec &b);
    double abs(const Vec &a);
    const Vec operator+ (const Vec &a, const Vec &b);
    const Vec operator- (const Vec &a, const Vec &b);
    const Vec operator* (const double a, const Vec&b);
    double operator* (const Vec &a, const Vec &b);   //dot product (unlike octave...)
    const Vec operator* (const Vec &a, const double b);
    const Vec operator/ (const Vec &a, const double b);
    std::ostream& operator << (std::ostream &os, const Vec& vv);

    class Mat {
        public:
        std::vector<std::vector<double>> mm ;
            
        Mat()=delete; 
        Mat(const int nc, const int nr, const double c=0);
        Mat(std::vector<std::vector<double>>);
        ////mat(std::initializer_list<double> list);
        Mat(const Mat &obj);
        Mat& operator=(const Mat & a);

        std::vector<double>& operator [] (const int i );
        const std::vector<double>& operator [] (const int i ) const ;

        Mat&      operator*= (const double a) ;
        Mat&      operator/= (const double a) ;
        const Mat operator+() const ;
        const Mat operator-() const ;

        Mat& operator+=(const Mat & a);
        Mat& operator-=(const Mat & a);

        Mat& operator*= (const Mat a) ;


        const Mat T() const; //transpose
        const Mat triangle() const ;
        double det() const ;
        double trace() const ;

    };
    
    std::ostream& operator << (std::ostream &os, const Mat& Mat);
    const Mat  operator+ (const Mat &a, const Mat &b);
    const Mat  operator- (const Mat &a, const Mat &b);
    const Mat  operator* (const Mat &a, const Mat &b);
    const Mat  operator* (const Mat &a, const double b);
    const Mat  operator* (const double a, const Mat &b);

    const Vec  operator* (const Mat &a, const Vec &b);
    const Vec  operator* (const Vec &a, const Mat &b);
    const Mat  operator/ (const Mat &a, const double b);

    //const matNd pow (const mat &a, const int n);

    class t_ans_ev_vec{
        public:
        std::vector<double> eigenValRe;
        std::vector<double> eigenValIm;
        //Mat eigenVecl;
        //Mat eigenVecr;
        std::vector<Vec> eigenVeclv;
        std::vector<Vec> eigenVecrv;
    };

    #ifdef vecNd_BLAS
    inline t_ans_ev_vec wrap_dgeev(const Mat &obj) {
        if(obj.mm.size()!=obj.mm[0].size()){
          std::cerr << "square matrix only is allowed." <<std::endl;
          exit(EXIT_FAILURE);
        }

        int VDIM=obj.mm.size();

        double *in = new double [VDIM*VDIM];
        double *eigenVecl= new double [VDIM*VDIM];
        double *eigenVecr= new double [VDIM*VDIM];

        for(int i = 0; i< VDIM; i++){
        for(int j = 0; j< VDIM; j++){
          in[i*VDIM+j] = obj[i][j];
          eigenVecl[i*VDIM+j] = 0.0 ;
          eigenVecr[i*VDIM+j] = 0.0 ;
        }
        }

        double *eigenValRe = new double [VDIM];
        double *eigenValIm = new double [VDIM];
        for(int i = 0; i< VDIM; i++){
          eigenValRe[i] = eigenValIm[i] = 0 ;
        }

        t_ans_ev_vec  ans;
        Mat copy = obj;
        LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'V', 'V', 
           VDIM, in, VDIM, eigenValRe, 
           eigenValIm, eigenVecl, VDIM,  eigenVecr, 
           VDIM);
          
        for(int i = 0; i< VDIM; i++){
          ans.eigenValRe.push_back(eigenValRe[i]);
          ans.eigenValIm.push_back(eigenValIm[i]);
        }
        for(int i = 0; i< VDIM; i++){
          ans.eigenVeclv.emplace_back(VDIM,0);
          ans.eigenVecrv.emplace_back(VDIM,0);
        }
        for(int i = 0; i< VDIM; i++){
          for(int j = 0; j< VDIM; j++){
            ans.eigenVeclv[j][i] = eigenVecl[i*VDIM+j];
            ans.eigenVecrv[j][i] = eigenVecr[i*VDIM+j];
          }
        }
        delete [] in;
        delete [] eigenVecl;
        delete [] eigenVecr;
        return ans;
    }
    #endif


    std::ostream& operator << (std::ostream &os, const t_ans_ev_vec& obj){
      int VDIM = obj.eigenValIm.size();
        for(int i=0; i<VDIM; i++){
            os << "Eigen val[ " << i << " ]= " << obj.eigenValRe[i] <<" + " << obj.eigenValIm[i] <<" *i" <<std::endl;//with pritty print 
        }
        os << "Eigen vec (left)" <<std::endl;
        for(int i=0; i<VDIM; i++){
            os  <<   (obj.eigenVeclv[i]) << std::endl ;
        }
        os << "Eigen vec (right)" <<std::endl;
        for(int i=0; i<VDIM; i++){
            os  <<   (obj.eigenVecrv[i]) ;
            if(i!=VDIM-1){
                os  <<  std::endl;
            }
        }
        return os;
    }


    ////defintions 
    inline Vec::Vec(const int dim, const double x){
        this->v=std::vector<double>(dim, x);
    }

    //inline vecNd<VDIM>::vecNd(const double x){
    //    for(int i = 0; i< size; ++i ){
    //        vec[i]=x;
    //    }
    //}

    inline Vec::Vec (const Vec & a) {
        v=a.v;
    }

    inline Vec::Vec(const std::vector<double>& vec_in){
        for(size_t i = 0; i< vec_in.size(); ++i ){
            v=vec_in;
        }
    }
    //template <int VDIM > 
    //inline vecNd<VDIM>::vecNd(std::initializer_list<double> list){
    //    if(list.size() != size){
    //        std::cerr << "warning: length of initializer list is not consistent!! " << std::endl;
    //    }
    //    auto itr = list.begin();
    //    for (int i = 0; i<size; ++i){
    //        vec[i]=*itr;
    //        itr++;
    //    }
    //}

    inline Vec& Vec::operator= (const Vec & a){
        this->v=a.v;
        return *this;
    }

    //
    inline Vec& Vec::operator+= (const Vec & a) {
        if(this->size() != a.size()){
          std::cerr << "vector sizes do not much" <<std::endl;
          exit(EXIT_FAILURE);
        }
        for(int i=0; i< this->size(); i++){
            this->v[i]+=a[i];
        }
        return *this;
    } 

    inline Vec& Vec::operator-= (const Vec & a) {
        if(this->size() != a.size()){
          std::cerr << "vector sizes do not much" <<std::endl;
          exit(EXIT_FAILURE);
        }
        for(int i=0; i< this->size(); i++){
            this->v[i]-=a.v[i];
        }
        return *this;
    } 

    inline Vec& Vec::operator*= (const double a) {
        for(int i=0; i< this->size(); i++){
            this->v[i]*=a;
        }
        return *this;
    } 

    inline Vec& Vec::operator/= (const double a) {
        double tmp = 1.0/a;
        return (*this)*=tmp;
    } 

    inline double Vec::dot (const Vec &a) const {
        if(this->size() != a.size()){
          std::cerr << "vector sizes do not much" <<std::endl;
          exit(EXIT_FAILURE);
        }
        double sum=0;
        for(int i=0; i < (this->size)(); ++i){
            sum+=a.v[i]*(this->v)[i];
        }
        return sum;
    }

    inline double Vec::abs () const {
        return sqrt(this->dot(*this));
    }

    inline double Vec::sum () const {
        double sum = 0.0;
        for(int i = 0; i< this->size(); ++i){
            sum+=v[i];
        }
        return sum;
    }

    inline double & Vec::operator [] (const int i ) {
        return this->v[i];
    }

    inline double Vec::operator [] (const int i ) const {
        return this->v[i];
    }


    //template <int VDIM > 
    //inline double & vecNd<VDIM>::at (const int i ) {
    //    if(i<0 or VDIM<=i){
    //        std::cerr << "Improper access" << std::endl;
    //        exit(EXIT_FAILURE) ;
    //    }
    //    return this->vec[i];
    //}

    //inline double vecNd<VDIM>::at (const int i ) const {
    //    if(i<0 or VDIM<=i){
    //        std::cerr << "Improper access" << std::endl;
    //        exit(EXIT_FAILURE) ;
    //    }
    //    return this->vec[i];
    //}

    inline double dot(const Vec &a, const Vec &b){
        return a.dot(b);
    }

    inline double abs(const Vec &a){
        return sqrt(dot(a, a));
    }

    std::ostream& operator << (std::ostream &os, const Vec& vv){
        os << vv.v ; //with prittyprint 
        return os;
    }

    inline const Vec operator + (const Vec &a, const Vec &b){
        return Vec(a)+=b;
    }

    inline const Vec operator - (const Vec &a, const Vec &b){
        return Vec(a)-=b;
    }

    inline const Vec Vec::operator + () const {
        return Vec(*this);
    }

    inline const Vec Vec::operator - () const {
        return Vec(*this)*=-1.0;
    }

    inline const Vec operator * (const double a, const Vec&b){
        return Vec(b)*=a;
    }

    inline double operator * (const Vec &a, const Vec &b){
        return dot(a,b);
    }

    inline const Vec operator * (const Vec &a, const double b){
        return Vec(a)*=b;
    }

    inline const Vec operator/ (const Vec &a, const double b){
        return Vec(a)/=b;
    }


    // Definitions regarding Mat start from here. 
    inline Mat::Mat(std::vector<std::vector<double>> v_in){ 
        mm=v_in;
    }

    inline Mat::Mat(const int m, const int n, const double c){
        mm=std::vector<std::vector<double>>(m, std::vector<double>(n, c));
    }

    //template<int VDIM>
    //inline matNd<VDIM>::matNd(std::initializer_list<double> list){
    //    if(list.size() != size*size){
    //        std::cerr << "warning: length of initializer list is not consistent!! " << std::endl;
    //    }
    //    auto itr = list.begin();
    //    for(int i = 0; i<VDIM; ++i){
    //        for(int j = 0; j<VDIM; j++){
    //            mat[i][j] = *itr;
    //            itr++;
    //        }
    //    }
    //}

    inline Mat::Mat(const Mat &obj){
        this->mm = obj.mm;
    }

    //template<int VDIM>
    //inline matNd<VDIM>::matNd(double x){
    //    for(int i =0 ; i < VDIM; ++i){                    
    //        for(int j =0 ; j < VDIM; ++j){                    
    //            mat[i][j] = x ;
    //        }
    //    }
    //}

    //template<int VDIM>
    //inline matNd<VDIM>::matNd(vecNd<VDIM> vec[VDIM]){
    //    for(int i =0 ; i < VDIM; ++i){                    
    //        for(int j =0 ; j < VDIM; ++j){                    
    //            mat[i][j] = vec[i][j] ;
    //        }
    //    }
    //}

    inline Mat& Mat::operator=(const Mat & a){
        this->mm = a.mm ;
        return *this;
    }

    inline std::vector<double>& Mat::operator [] (const int i ){
        return this->mm[i];
    }
  
    inline const std::vector<double>&  Mat::operator [] (const int i ) const {
        return this->mm[i];
    }

    inline const Mat Mat::operator+() const {
        return Mat(*this);
    }
    
    inline const Mat Mat::operator-() const {
        return Mat(*this)*=-1.0;
    }
    
    inline Mat& Mat::operator+=(const Mat & a){
        if(this->mm.size() != a.mm.size()){
          std::cerr << "Matrix sizes do not much" <<std::endl;
          exit(EXIT_FAILURE);
        }
        if(this->mm[0].size() != a.mm[0].size()){
          std::cerr << "Matrix sizes do not much" <<std::endl;
          exit(EXIT_FAILURE);
        }

        for(size_t i =0 ; i < this->mm.size(); ++i){
            for(size_t j =0 ; j < this->mm[i].size(); ++j){
                mm[i][j] += a[i][j] ;
            }
        }
        return *this;
    }

    inline Mat& Mat::operator-=(const Mat & a){
        return *this+=(-a);
    }

    Mat& Mat::operator*= (const double a) {
        for(size_t i =0 ; i < this->mm.size(); ++i){
            for(size_t j =0 ; j < this->mm[i].size(); ++j){
                mm[i][j]*=a ;
            }
        }
        return *this;
    } 

    Mat& Mat::operator/= (const double a) {
        return (*this)*=(1.0/a);
    } 

    inline Mat& Mat::operator*= (const Mat a) {
        Mat results(this->mm.size(),a.mm[0].size(), 0.0); 
        for(size_t i =0 ; i < this->mm.size(); ++i){
            for(size_t j =0 ; j < a.mm[0].size(); ++j){
               for(size_t k =0 ; k < this->mm[0].size(); ++k){
                    results.mm[i][j]+=mm[i][k]*a[k][j];
               }
            }
        }
        *this=results;
        return *this;

    } 


    inline const Mat Mat::T() const{
        Mat result(this->mm[0].size(),this->mm.size());
        for(size_t i =0 ; i < this->mm.size() ; ++i){
            for(size_t j =0 ; j < this->mm[0].size(); ++j){
                result[j][i] = (*this)[i][j] ;
            }
        }
        return result;
    }



    std::ostream& operator << (std::ostream &os, const Mat& mat){
        os << mat.mm ; //with pritty print 
        return os;
    }

    inline const Mat  operator+ (const Mat &a, const Mat &b){
        return Mat(a)+=b;
    }

    inline const Mat  operator- (const Mat &a, const Mat &b){
        return Mat(a)-=b;
    }

    inline const Mat  operator* (const Mat &a, const Mat &b){
        return Mat(a)*=b;
    }

    //template<int VDIM>
    //inline const matNd<VDIM>  operator* (const matNd<VDIM> &a, const double b){
    //    return matNd<VDIM>(a)*=b;
    //}

    //template<int VDIM>
    //inline const matNd<VDIM>  operator* (const double a, const matNd<VDIM> &b){
    //    return matNd<VDIM>(b)*=a;
    //}

    inline const Vec  operator* (const Mat &a, const Vec &b){
        Vec result(a.mm.size(), 0.0);
        for(size_t i=0; i < a.mm.size(); i++){
            for(size_t j=0; j < a.mm[0].size(); j++){
                result[i]+=a[i][j]*b[j];
            }
        }
        return result;
    }

    inline const Vec  operator* (const Vec &a, const Mat &b){
        Vec result(b.mm[0].size(), 0.0);
        for(size_t i=0; i < b.mm[0].size(); i++){
            for(size_t j=0; j < b.mm.size(); j++){
                result[i]+=a[j]*b[j][i];
            }
        }
        return result;
    }


    //template<int VDIM>
    //inline const matNd<VDIM> pow (const matNd<VDIM> &a, const int n){
    //    if(n == 0){
    //        matNd<VDIM> E;
    //        return E;
    //    }
    //    if(n < 0){
    //        std::cerr <<"invalid power index. Index must be non-negative integer" << std::endl;
    //        exit(EXIT_FAILURE);
    //    }
    //    //for n>1
    //    if(n%2 == 0){
    //        return  pow(a*a, n/2) ;
    //    }
    //    else /*if(n%2 ==1)*/{
    //        return  a*pow(a*a, (n)/2) ;
    //    }
    //}

    inline const Mat  operator* (const Mat  &a, const double b){
        return Mat(a)*=b;
    }
    inline const Mat  operator* (const double b,const Mat  &a){
        return Mat(a)*=b;
    }


    inline const Mat  operator/ (const Mat  &a, const double b){
        return Mat(a)/=b;
    }

    //template<int VDIM>
    //inline const matNd<VDIM> operator+ (const matNd<VDIM> &a){
    //    return a;
    //}

    //template<int VDIM>
    //inline const matNd<VDIM> operator- (const matNd<VDIM> &a){
    //    return matNd<VDIM>(a)*=-1.0;
    //}
    //    
    //// outerproduct is defined for N=VDIM only. this should be practical enough.
    //
    //inline const vecNd<3> cross(const vecNd<3> &a, const vecNd<3> &b){
    //    int VDIM=3;
    //    vecNd<3> result(0.0);
    //    for (int i=0 ; i <VDIM ; ++i){
    //        result[i]+=a[(i+1)%VDIM]*b[(i+2)%VDIM];
    //    }
    //    for (int i=0 ; i <VDIM ; ++i){
    //        result[i]-=a[(i+2)%VDIM]*b[(i+1)%VDIM];
    //    }
    //    return result;
    //}

    double Mat::trace () const {
        double res=0;
        if(this->mm.size() != this->mm[0].size()){
          std::cerr << "square matrix only is allowed." <<std::endl;
          exit(EXIT_FAILURE);
        }
        for(size_t i = 0; i < mm.size(); i++){
          res+=mm[i][i];
        }
        return res;
    }

    const Mat Mat::triangle () const {
        Mat result=(*this);
        if(this->mm.size() != this->mm[0].size()){
          std::cerr << "square matrix only is allowed." <<std::endl;
          exit(EXIT_FAILURE);
        }

        int VDIM = mm.size();
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

    //template<int VDIM>
    //inline const matNd<VDIM> triangle (const matNd<VDIM> & a )  {
    //    return a.triangle();
    //}

    inline double Mat::det () const {
        Mat tmp(this->triangle());
        double result=1.0;
        int VDIM = mm.size();
        for(int i=0; i<VDIM; i++){
            result*=tmp[i][i];
        }
        return result;
    }

    inline double det (Mat const &a){
        return a.det();
    }

}


#endif //for include guard
