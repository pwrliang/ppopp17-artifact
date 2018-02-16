//
// Created by liang on 2/15/18.
//

#ifndef GROUTE_REGISTRY_H
#define GROUTE_REGISTRY_H

#include <map>
#include <string>
//
//class RunnerRegistry {
//private:
//    RunnerRegistry() {}
//
//    Map;
//    Map m_;
//public:
//    typedef int (*KernelRunner)();
//
//    typedef std::map<std::string, KernelRunner> Map;
//
//    KernelRunner runner(const string &name) {
//        return m_[name];
//    }
//
//    Map &runners() { return m_; }
//
//    static RunnerRegistry *Get();
//};
//
//struct RunnerRegistrationHelper {
//    RunnerRegistrationHelper(RunnerRegistry::KernelRunner k, const char *name) {
//        RunnerRegistry::Get()->runner().insert(make_pair(name, k));
//    }
//};
//
//template<class C, class K, class V, class D>
//struct KernelRegistrationHelper {
//    KernelRegistrationHelper(const char *name, MaiterKernel <V, D> *maiter) {
//        KernelRegistry::Map &kreg = KernelRegistry::Get()->kernels();
//
//        CHECK(kreg.find(name) == kreg.end());
//        kreg.insert(make_pair(name, new KernelInfoT<C, K, V, D>(name, maiter)));
//    }
//};
//
//struct RunnerRegistrationHelper {
//    RunnerRegistrationHelper(RunnerRegistry::KernelRunner k, const char *name) {
//        RunnerRegistry::Get()->runners().insert(make_pair(name, k));
//    }
//};
//
//#define REGISTER_KERNEL(klass)\
//  static KernelRegistrationHelper<klass> k_helper_ ## klass(#klass);
//
//#define REGISTER_METHOD(klass, method)\
//  static MethodRegistrationHelper<klass> m_helper_ ## klass ## _ ## method(#klass, #method, &klass::method);
//
//#define REGISTER_RUNNER(r)\
//  static RunnerRegistrationHelper r_helper_ ## r ## _(&r, #r);

#endif //GROUTE_REGISTRY_H
