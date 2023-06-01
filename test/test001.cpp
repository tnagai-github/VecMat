#include<iostream>
#include <iomanip>
#include<vector>
#include "cxx-prettyprint/prettyprint.hpp"
#include "vecNd.hpp"
#include "vec.hpp"

#include<numeric>

#ifdef vecNd_BLAS
#include <lapacke.h>
#include <cblas.h>
#endif

#include "gtest/gtest.h"

using namespace VecMat;

class VecNdTest : public ::testing::Test {
  protected:
    class vecNd<3> vv000;
    class vecNd<3> vv001 = vecNd<3>(1.0);
    class vecNd<3> vv002;
    class vecNd<3> vv003;
    class vecNd<3> vv_ex;
    class vecNd<3> vv_ey;
    class vecNd<3> vv_ez;
    virtual void SetUp(){
      double darray1[3] = {0.5, -1, 2.0};
      vv002=vecNd<3>(darray1);
      vv003=vv001;
      vv003[2] = 0;
      vv_ex[0]=1;
      vv_ey[1]=1;
      vv_ez[2]=1;
    }
};

TEST_F(VecNdTest, Basics000_initialization){
  EXPECT_EQ(vecNd<3>(0.0), vv000) ;
  EXPECT_EQ(vecNd<3>(1.0), vv001) ;
  EXPECT_EQ( 0.5, vv002[0]) ;
  EXPECT_EQ(-1.0, vv002[1]) ;
  EXPECT_EQ( 2.0, vv002[2]) ;
}

TEST_F(VecNdTest, Basics001){
  EXPECT_EQ(vecNd<3>(-1.0), -vv001) ;
  EXPECT_EQ(vecNd<3>(-3.0), vv001*(-3.0)) ;
  EXPECT_EQ(vecNd<3>(3.0), (3.0)*vv001) ;
  EXPECT_EQ(vecNd<3>(-1.0), vv001/(-1.0)) ;
  EXPECT_EQ(vecNd<3>(0.5), vv001/=(2.0)) ;
  EXPECT_EQ(vecNd<3>(-1.0), vv001*=(-2.0)) ;
}

TEST_F(VecNdTest, Basics002){
  EXPECT_DOUBLE_EQ(sqrt(3), vv001.abs()) ;
  EXPECT_DOUBLE_EQ(sqrt(2), abs(vv003)) ;
  EXPECT_DOUBLE_EQ(3, vv001*vv001) ;
  EXPECT_EQ(vv_ez, cross(vv_ex, vv_ey)) ;
  EXPECT_EQ(vv_ex, cross(vv_ey, vv_ez)) ;
  EXPECT_EQ(vv_ey, cross(vv_ez, vv_ex)) ;
  EXPECT_DOUBLE_EQ(-0.5, (vv002*(vv_ex+vv_ey))) ;
}

class MatNdTest : public ::testing::Test {
  protected:
    class matNd<3> mat11;
    class matNd<3> mat12 = matNd<3>(1);
    class matNd<3> mat13 {9,8,7,6,5,4,3,2,1};
    class matNd<3> mat14 = matNd<3>(0.0);
    virtual void SetUp(){
    }
};

TEST_F(MatNdTest, Basics000_initialization){
  EXPECT_DOUBLE_EQ(1.0, mat11[0][0]) ;
  EXPECT_DOUBLE_EQ(0.0, mat11[0][1]) ;
  EXPECT_DOUBLE_EQ(0.0, mat11[0][2]) ;
  EXPECT_DOUBLE_EQ(0.0, mat11[1][0]) ;
  EXPECT_DOUBLE_EQ(1.0, mat11[1][1]) ;
  EXPECT_DOUBLE_EQ(0.0, mat11[1][2]) ;
  EXPECT_DOUBLE_EQ(0.0, mat11[2][0]) ;
  EXPECT_DOUBLE_EQ(0.0, mat11[2][1]) ;
  EXPECT_DOUBLE_EQ(1.0, mat11[2][2]) ;
  EXPECT_DOUBLE_EQ(1.0, mat12[0][0]) ;
  EXPECT_DOUBLE_EQ(1.0, mat12[0][1]) ;
  EXPECT_DOUBLE_EQ(1.0, mat12[0][2]) ;
  EXPECT_DOUBLE_EQ(1.0, mat12[1][0]) ;
  EXPECT_DOUBLE_EQ(1.0, mat12[1][1]) ;
  EXPECT_DOUBLE_EQ(1.0, mat12[1][2]) ;
  EXPECT_DOUBLE_EQ(1.0, mat12[2][0]) ;
  EXPECT_DOUBLE_EQ(1.0, mat12[2][1]) ;
  EXPECT_DOUBLE_EQ(1.0, mat12[2][2]) ;
  EXPECT_DOUBLE_EQ(9.0, mat13[0][0]) ;
  EXPECT_DOUBLE_EQ(8.0, mat13[0][1]) ;
  EXPECT_DOUBLE_EQ(7.0, mat13[0][2]) ;
  EXPECT_DOUBLE_EQ(6.0, mat13[1][0]) ;
  EXPECT_DOUBLE_EQ(5.0, mat13[1][1]) ;
  EXPECT_DOUBLE_EQ(4.0, mat13[1][2]) ;
  EXPECT_DOUBLE_EQ(3.0, mat13[2][0]) ;
  EXPECT_DOUBLE_EQ(2.0, mat13[2][1]) ;
  EXPECT_DOUBLE_EQ(1.0, mat13[2][2]) ;
}

TEST_F(MatNdTest, Basics001){
  EXPECT_DOUBLE_EQ(-9.0, (-mat13)[0][0]) ;
  EXPECT_DOUBLE_EQ(-8.0, (-mat13)[0][1]) ;
  EXPECT_DOUBLE_EQ(-7.0, (-mat13)[0][2]) ;
  EXPECT_DOUBLE_EQ(-6.0, (-mat13)[1][0]) ;
  EXPECT_DOUBLE_EQ(-5.0, (-mat13)[1][1]) ;
  EXPECT_DOUBLE_EQ(-4.0, (-mat13)[1][2]) ;
  EXPECT_DOUBLE_EQ(-3.0, (-mat13)[2][0]) ;
  EXPECT_DOUBLE_EQ(-2.0, (-mat13)[2][1]) ;
  EXPECT_DOUBLE_EQ(-1.0, (-mat13)[2][2]) ;
  EXPECT_EQ(mat14, mat13-mat13) ;
}
//    std::cout << std::endl;
//    std::cout << std::endl;
//    std::cout << "***************************"              <<std::endl;
//    std::cout << "matNd<VDIM>"              <<std::endl;
//    std::cout << "VDIM =3"                    <<std::endl;
//    std::cout << "mat11:" << mat11            <<std::endl;
//    std::cout << "mat12:" << mat12            <<std::endl;
//    std::cout << "mat13:" << mat13            <<std::endl;
//    std::cout << "mat14:" << mat14            <<std::endl;
//    std::cout << "+mat14:" << +mat14            <<std::endl;
//    std::cout << "-mat14:" << -mat14            <<std::endl;
//    std::cout << "mat11[1][2]:" <<mat11[1][2]      <<std::endl;
//    std::cout << "mat11[2][2]:" <<mat11[2][2]      <<std::endl;
//    std::cout << "(mat11+=mat12):" << (mat11+=mat12)   <<std::endl;
//    std::cout << "(mat11-=mat12):" << (mat11-=mat12)   <<std::endl;
//    std::cout << "(mat11+mat12):" << (mat11+mat12)   <<std::endl;
//    std::cout << "(mat11-mat12):" << (mat11-mat12)   <<std::endl;
//    std::cout << "mat11*=2.0: "   << (mat11*=2.0)     <<std::endl;
//    std::cout << "mat11/=2.0: "   << (mat11/=2.0)     <<std::endl;
//    std::cout << "mat11*2.0: "    << (mat11*2.0)     <<std::endl;
//    std::cout << "mat11/2.0: "    << (mat11/2.0)     <<std::endl;
//    std::cout << "mat13: "        <<  (mat13)         <<std::endl;
//    std::cout << "mat13.T(): "    << (mat13.T())      <<std::endl;
//    std::cout << "mat13*mat13: "    << (mat13*mat13)      <<std::endl;
//    std::cout << "***************************"              <<std::endl;
//    std::cout << "product of matrix and vector is also possible"  <<std::endl;
//    std::cout << "mat13*vv12:"    << (mat13*vv12)     <<std::endl;
//    std::cout << "v12*mat13.T()"  << (vv12*mat13.T()) <<std::endl;
//
//    std::cout << "***************************"              <<std::endl;
//    std::cout << std::endl;
//    std::cout << std::endl;
//    std::cout << "VDIM =4"              <<std::endl;
//    std::cout << "mat21:" << mat21            <<std::endl;
//    std::cout << "mat22:" << mat22            <<std::endl;
//    std::cout << "mat23:" << mat23            <<std::endl;
//    std::cout << "mat21[1][2]:"  << mat21[1][2]      <<std::endl;
//    std::cout << "mat21[2][2]:"  << mat21[2][2]      <<std::endl;
//    std::cout << "(mat21+=mat22):" << (mat21+=mat22)   <<std::endl;
//    std::cout << "(mat21-=mat22):" << (mat21-=mat22)   <<std::endl;
//    std::cout << "(mat21+mat22):" << (mat21+mat22)   <<std::endl;
//    std::cout << "(mat21-mat22):" << (mat21-mat22)   <<std::endl;
//    std::cout << "mat21*=2.0: "   << (mat21*=2.0)     <<std::endl;
//    std::cout << "mat21/=2.0: "   << (mat21/=2.0)     <<std::endl;
//    std::cout << "mat21*2.0: "    << (mat21*2.0)     <<std::endl;
//    std::cout << "mat21/2.0: "    << (mat21/2.0)     <<std::endl;
//    std::cout << "mat23: "        <<  (mat23)         <<std::endl;
//    std::cout << "mat23.T(): "    << (mat23.T())      <<std::endl;
//    std::cout << "mat23*mat23: "    << (mat23*mat23)      <<std::endl;
//    std::cout << "***************************"              <<std::endl;
//    std::cout << "product of matrix and vector is also possible"  <<std::endl;
//    std::cout << "mat23*vv22:"    << (mat23*vv22)     <<std::endl;
//    std::cout << "v22*mat23.T():"  << (vv22*mat23.T()) <<std::endl;
//    std::cout << "***************************"              <<std::endl;
//
//
//
//
//    std::cout << std::endl;
//    std::cout << std::endl;
//    std::cout << "***************************"              <<std::endl;
//    std::cout << "Some rotation matrices are predefined"   <<std::endl;
//    std::cout << "rot_by_x(45/360.*2*M_PI): " << rot_by_x(45/360.*2*M_PI) <<std::endl;
//    std::cout << "rot_by_y(45/360.*2*M_PI): " << rot_by_y(45/360.*2*M_PI) <<std::endl;
//    std::cout << "rot_by_z(45/360.*2*M_PI): " << rot_by_z(45/360.*2*M_PI) <<std::endl;
//    std::cout << "rot_by_z(45 deg) *  [1, 0, 0] : " << rot_by_z(45/360.*2*M_PI)*vecNd<3>{1,0,0} <<std::endl;
//    std::cout << "rot_by_z(45 deg) *  [0, 1, 0] : " << rot_by_z(45/360.*2*M_PI)*vecNd<3>{0,1,0} <<std::endl;
//    std::cout << "rot_by_x(45 deg) *  [0, 1, 0] : " << rot_by_x(45/360.*2*M_PI)*vecNd<3>{0,1,0} <<std::endl;
//    std::cout << "rot_by_x(45 deg) *  [0, 0, 1] : " << rot_by_x(45/360.*2*M_PI)*vecNd<3>{0,0,1} <<std::endl;
//    std::cout << "rot_by_y(45 deg) *  [0, 0, 1] : " << rot_by_y(45/360.*2*M_PI)*vecNd<3>{0,0,1} <<std::endl;
//    std::cout << "rot_by_y(45 deg) *  [1, 0, 0] : " << rot_by_y(45/360.*2*M_PI)*vecNd<3>{1,0,0} <<std::endl;
//    std::cout << "rot_mat(45/360.*2*M_PI): " << rot_mat(45/360.*2*M_PI) <<std::endl;
//    std::cout << "rot_mat(45 deg) * [1, 0]: " << rot_mat(45/360.*2*M_PI)*vecNd<2>{1,0}  <<std::endl;
//    std::cout << "rot_mat(45 deg) * [0, 1]: " << rot_mat(45/360.*2*M_PI)*vecNd<2>{0,1}  <<std::endl;
//    std::cout << "***************************"              <<std::endl;
//
//
//
//
//    std::cout << std::endl;
//    std::cout << std::endl;
//    std::cout << "***************************"              <<std::endl;
//    std::cout << "Very basic matrix operation"   <<std::endl;
//    std::cout << "triangle(mat13): " << triangle(mat13) <<std::endl;
//    std::cout << "(mat13.triangle()): " << (mat13.triangle()) <<std::endl;
//    std::cout << "det(mat13): " << det(mat13) <<std::endl;
//    std::cout << "(mat13.det()): " << (mat13.det()) <<std::endl;
//    std::cout << "triangle(mat23): " << triangle(mat23) <<std::endl;
//    std::cout << "(mat23.triangle()): " << (mat23.triangle()) <<std::endl;
//    std::cout << "det(mat23): " << det(mat23) <<std::endl;
//    std::cout << "(mat23.det()): " << (mat23.det()) <<std::endl;
//    std::cout << "(pow(mat23,0): " << (pow(mat23,0)) <<std::endl;
//    std::cout << "(pow(mat23,1): " << (pow(mat23,1)) <<std::endl;
//    std::cout << "(pow(mat23,2): " << (pow(mat23,2)) <<std::endl;
//    std::cout << "***************************"       <<std::endl;
//
//
//
//// as operators are overloaded, STL related function is usable
//    std::vector< class matNd<3>> mat_array1(3);
//
//    std::cout << mat_array1 << std::endl;
//    
//    class matNd<3> init(0.0);
//    auto sum1 = std::accumulate(mat_array1.begin()+1, mat_array1.end(), mat_array1[0]);
//
//    std::cout << "sum1: " << sum1 << std::endl;
//
//
//    class matNd<4> mat23v2(mat23);
//    double wi[4];
//    double wr[4];
//
//    //double vl[4][4];
//    //double vr[4][4];
//    class matNd<4> vl2 ;
//    class matNd<4> vr2 ;
//
//#ifdef vecNd_BLAS
//	//direct call of blas
//    LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', 
//                         4, mat23v2.p, 4, wr, 
//                          wi, vl2.p, 4,  vr2.p, 
//                          4 );
//	//called with wrapper 
//    auto ans2 = wrap_dgeev(mat23);
//
//    std::cout << " *wrapper's results(dgeev)*" << std::endl;
//    std::cout << ans2.eigenValRe << std::endl;
//    std::cout << ans2.eigenValIm << std::endl;
//    std::cout << ans2.eigenVecl << std::endl;
//    std::cout << ans2.eigenVecr << std::endl;
//    std::cout << ans2 << std::endl;
//    std::cout << "***********" << std::endl;
//
//
//    double init_array3[4][4]= {{2,5,0,1},{5,5,1,1},{0,1,4,0},{1,1,0,1}};
//    class matNd<4> mat24(init_array3);
//    std::cout << "mat24" << mat24 << std::endl;
//
//    std::cout << "***********" << std::endl;
//    std::cout << " *is_symetrici(mat24):  " << is_symetric(mat24) << std::endl;
//    std::cout << " *is_positive_definite_dsy(mat24):  " << is_positive_definite_dsy(mat24) << std::endl;
//    std::cout << " *calc_inverse(mat24):  "  << calc_inverse(mat24) << std::endl;
//    std::cout << " *mat24*calc_inverse(mat24):  "  << mat24*calc_inverse(mat24) << std::endl;
//    std::cout << "***********" << std::endl;
//
//    ans2 = wrap_dgeev(mat24);
//    std::cout << " *wrapper's results(dgeev, mat24)*" << std::endl;
//    std::cout << ans2.eigenValRe << std::endl;
//    std::cout << ans2.eigenValIm << std::endl;
//    std::cout << ans2.eigenVecl << std::endl;
//    std::cout << ans2.eigenVecr << std::endl;
//    std::cout << ans2 << std::endl;
//    std::cout << "***********" << std::endl;
//
//
//    ans2 = wrap_dsyev(mat24);
//    std::cout << " *wrapper's results(dsyev, mat24)*" << std::endl;
//    std::cout << ans2.eigenValRe << std::endl;
//    std::cout << ans2.eigenValIm << std::endl;
//    std::cout << ans2.eigenVecl << std::endl;
//    std::cout << ans2.eigenVecr << std::endl;
//    std::cout << ans2 << std::endl;
//    std::cout << "***********" << std::endl;
//    std::cout << "*conform *" << std::endl;
//    std::cout << ans2.eigenVecl*mat24*ans2.eigenVecr.T() << std::endl;
//    std::cout << ans2.QT()*mat24*ans2.Q() << std::endl;
//    std::cout << "***********" << std::endl;
//    
//	// to collect the results obtained by a direct call
//    class matNd<4> tmp(vr2.T()); 
//    class vecNd<4>  eigenv0(tmp[0]);
//    class vecNd<4>  eigenv1(tmp[1]);
//    class vecNd<4>  eigenv2(tmp[2]);
//    class vecNd<4>  eigenv3(tmp[3]);
//
//
//    std::cout << " *direct call of lapack *" << std::endl;
//    std::cout << eigenv0 << std::endl;
//    std::cout << eigenv1 << std::endl;
//    std::cout << eigenv2 << std::endl;
//    std::cout << eigenv3 << std::endl;
//
//    std::cout << "******" << std::endl;
//
//    std::cout << " *check Mv=av **" << std::endl;
//    std::cout << mat23*eigenv0 << std::endl;
//    std::cout << wr[0]*eigenv0 << std::endl;
//
//    std::cout << mat23*eigenv1 << std::endl;
//    std::cout << wr[1]*eigenv1 << std::endl;
//
//    std::cout << mat23*eigenv2 << std::endl;
//    std::cout << wr[2]*eigenv2 << std::endl;
//
//    std::cout << mat23*eigenv3 << std::endl;
//    std::cout << wr[3]*eigenv3 << std::endl; //return of vr is column major !!
//
//
//    
//
//    double ans;
//    ans = cblas_ddot(vv11.size, vv11.p, 1, vv12.p,1);
////    double cblas_ddot(const int N, const double *X, const int incX,
////                  const douggble *Y, const int incY);
//    std::cout << ans << std::endl;
//    std::cout << vv11*vv12 << std::endl;
//#endif
//
//
//
//    Vec vecA(3,1.0/sqrt(3.0));
//    Vec vecB(std::vector<double>{1,-1,1});
//    vecA[1] = 2;
//    Vec vecC(vecA);
//
//    std::cout << "vecA          " << vecA           << std::endl;
//    std::cout << "vecA          " << vecA           << std::endl;
//    std::cout << "vecA.dot(vecA)" << vecA.dot(vecA) << std::endl;
//    std::cout << "vecA.abs()    " << vecA.abs()     << std::endl;
//    std::cout << "vecA.sum()    " << vecA.sum()     << std::endl;
//    std::cout << "vecA*vecB     " << vecA*vecB      << std::endl;
//    std::cout << "(vecB)        " << (vecB)         << std::endl;
//    std::cout << "(vecB*=5)     " << (vecB*=5)      << std::endl;
//    std::cout << "(vecB/=5)     " << (vecB/=5)      << std::endl;
//    std::cout << "(vecB-=vecA)  " << (vecB-=vecA)   << std::endl;
//    std::cout << "(+vecB)       " << (+vecB)        << std::endl;
//    std::cout << "(-vecB)       " << (-vecB)        << std::endl;
//
//    Mat matA(std::vector<std::vector<double>>{std::vector<double>{1,2,3}, std::vector<double>{4,5,6}, std::vector<double>{7,8,9}, std::vector<double>{10,11,12}});
//
//    Mat matB(std::vector<std::vector<double>>{std::vector<double>{1,1,1}, std::vector<double>{2,2,2}, std::vector<double>{3,3,3}, std::vector<double>{12,11,10}});
//    Mat matC(matB);
//    Mat matD=(matB);
//    //Mat matE=(std::vector<std::vector<double>>{std::vector<double>{1,1,1}, std::vector<double>{2,3,2}, std::vector<double>{3,3,4}});
//    Mat matE=(std::vector<std::vector<double>>{std::vector<double>{1,1}, std::vector<double>{1,2}});
//    matD[1][2]+=100;
//    std::cout << "matA          " << matA           << std::endl;
//    std::cout << "matB          " << matB           << std::endl;
//    std::cout << "matC          " << matC           << std::endl;
//    std::cout << "matC[0][2]    " << matC[0][2]    << std::endl;
//    matC[0][2] = 999 ;
//    std::cout << "matC[0][2]    " << matC[0][2]    << std::endl;
//    std::cout << "matC          " << matC          << std::endl;
//    std::cout << "matD          " << matD          << std::endl;
//    std::cout << "matD+=matC    " << (matD+=matC)    << std::endl;
//    std::cout << "+matD         " << (+matD)    << std::endl;
//    std::cout << "-matD         " << (-matD)    << std::endl;
//    std::cout << "matD.T()         " << (matD.T())    << std::endl;
//    std::cout << "matD*matD.T()    " << (matD*matD.T())    << std::endl;
//    std::cout << "matD*matD.T()    " << (matD*matD.T())    << std::endl;
//    std::cout << "matD*vecB        " << (matD*vecB)    << std::endl;
//    std::cout << "vecB*matD        " << (vecB*matD.T())    << std::endl;
//    std::cout << "matD/100.0       " << (matD/100.0)    << std::endl;
//    std::cout << "matD*10.0        " << (matD*10.0)    << std::endl;
//    std::cout << "10.0*matD*10.0  " << (10.0*matD*10.0)    << std::endl;
//    std::cout << "matE             " <<matE           << std::endl;
//    std::cout << "matE.trace() " <<matE.trace()   << std::endl;
//    std::cout << "matE.triangle() " <<matE.triangle()   << std::endl;
//    std::cout << "matE.det()      " <<matE.det()   << std::endl;
//
//    std::cout << "**************************"    << std::endl;
//#ifdef vecNd_BLAS
//    auto ans3=wrap_dgeev(matE) ;
//    std::cout << "Eigen vectors \n " << ans3   << std::endl;
//   
//    //check
//    std::cout << "Left vectors"    << std::endl;
//    for(size_t i = 0; i<matE.mm.size(); i++){
//      std::cout << " " << (ans3.eigenVeclv[i]*ans3.eigenValRe[i])   << std::endl;
//      std::cout << " " << (ans3.eigenVeclv[i]*matE)   << std::endl;
//    }
//    std::cout << "Right vectors"    << std::endl;
//    for(size_t i = 0; i<matE.mm.size(); i++){
//      std::cout << " " << (ans3.eigenVecrv[i]*ans3.eigenValRe[i])   << std::endl;
//      std::cout << " " << (matE*ans3.eigenVecrv[i])   << std::endl;
//    }
//#endif
//}
