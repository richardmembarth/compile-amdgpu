#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <sstream>

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

void store_file(const std::string& filename, const std::string& str);
std::string load_file(const std::string& filename);
std::string emit_gcn(const std::string &filename, const std::string& cpu, int opt);

int main(int argc, char** argv) {
    std::string filename;
    if (argc == 2)
        filename = argv[1];
    else
        error("usage: % 'llvmir.amdgpu'", argv[0]);

    emit_gcn(filename, "gfx803" /*arch*/, 3 /*opt*/);

    return EXIT_SUCCESS;
}

void store_file(const std::string& filename, const std::string& str) {
    std::ofstream dst_file(filename);
    if (!dst_file)
        error("Can't open destination file '%'", filename);
    dst_file << str;
    dst_file.close();
}

std::string load_file(const std::string& filename) {
    std::ifstream src_file(filename);
    if (!src_file.is_open())
        error("Can't open source file '%'", filename);

    return std::string(std::istreambuf_iterator<char>(src_file), (std::istreambuf_iterator<char>()));
}

static std::string get_ocml_config(int target) {
    std::string config = R"(
        ; Module anydsl ocml config
        @__oclc_finite_only_opt = addrspace(4) constant i8 0
        @__oclc_unsafe_math_opt = addrspace(4) constant i8 0
        @__oclc_daz_opt = addrspace(4) constant i8 0
        @__oclc_correctly_rounded_sqrt32 = addrspace(4) constant i8 0
        @__oclc_ISA_version = addrspace(4) constant i32 )";
    return config + std::to_string(target);
}

bool llvm_initialized = false;
std::string emit_gcn(const std::string &filename, const std::string& cpu, int opt) {
    if (!llvm_initialized) {
        std::vector<const char*> c_llvm_args;
        std::vector<std::string> llvm_args = { "gcn", "-opt-bisect-limit=-1" };
        for (auto &str : llvm_args)
            c_llvm_args.push_back(str.c_str());
        llvm::cl::ParseCommandLineOptions(c_llvm_args.size(), c_llvm_args.data(), "AnyDSL gcn JIT compiler\n");

        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUAsmPrinter();
        llvm_initialized = true;
    }

    const std::string& program = load_file(filename);

    llvm::LLVMContext llvm_context;
    llvm::SMDiagnostic diagnostic_err;
    std::unique_ptr<llvm::Module> llvm_module = llvm::parseIR(llvm::MemoryBuffer::getMemBuffer(program)->getMemBufferRef(), diagnostic_err, llvm_context);

    if (!llvm_module) {
        std::string stream;
        llvm::raw_string_ostream llvm_stream(stream);
        diagnostic_err.print("", llvm_stream);
        error("Parsing IR file %: %", filename, llvm_stream.str());
    }

    auto triple_str = llvm_module->getTargetTriple();
    std::string error_str;
    auto target = llvm::TargetRegistry::lookupTarget(triple_str, error_str);
    llvm::TargetOptions options;
    options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    options.NoTrappingFPMath = true;
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(triple_str, cpu, "-trap-handler" /* attrs */, options, llvm::Reloc::PIC_, llvm::CodeModel::Small, llvm::CodeGenOpt::Aggressive));

    // link ocml.amdgcn and ocml config
    std::string ocml_file = "/opt/rocm/lib/ocml.amdgcn.bc";
    if (cpu.compare(0, 3, "gfx"))
        error("Expected gfx ISA, got %", cpu);
    std::string ocml_config = get_ocml_config(std::stoi(&cpu[3 /*"gfx"*/]));
    std::unique_ptr<llvm::Module> ocml_module(llvm::parseIRFile(ocml_file, diagnostic_err, llvm_context));
    if (ocml_module == nullptr)
        error("Can't create ocml module for '%'", ocml_file);
    std::unique_ptr<llvm::Module> config_module = llvm::parseIR(llvm::MemoryBuffer::getMemBuffer(ocml_config)->getMemBufferRef(), diagnostic_err, llvm_context);
    if (config_module == nullptr)
        error("Can't create ocml config module");

    // override data layout with the one coming from the target machine
    llvm_module->setDataLayout(machine->createDataLayout());
    ocml_module->setDataLayout(machine->createDataLayout());
    config_module->setDataLayout(machine->createDataLayout());

    llvm::Linker linker(*llvm_module.get());
    if (linker.linkInModule(std::move(config_module), llvm::Linker::Flags::None))
        error("Can't link config into module");
    if (linker.linkInModule(std::move(ocml_module), llvm::Linker::Flags::LinkOnlyNeeded))
        error("Can't link ocml into module");

    llvm::legacy::FunctionPassManager function_pass_manager(llvm_module.get());
    llvm::legacy::PassManager module_pass_manager;

    module_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));
    function_pass_manager.add(llvm::createTargetTransformInfoWrapperPass(machine->getTargetIRAnalysis()));

    llvm::PassManagerBuilder builder;
    builder.OptLevel = opt;
    builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, 0, false);
    machine->adjustPassManager(builder);
    builder.populateFunctionPassManager(function_pass_manager);
    builder.populateModulePassManager(module_pass_manager);

    machine->Options.MCOptions.AsmVerbose = true;

    llvm::SmallString<0> outstr;
    llvm::raw_svector_ostream llvm_stream(outstr);

    //machine->addPassesToEmitFile(module_pass_manager, llvm_stream, nullptr, llvm::TargetMachine::CGFT_AssemblyFile, true);
    machine->addPassesToEmitFile(module_pass_manager, llvm_stream, nullptr, llvm::TargetMachine::CGFT_ObjectFile, true);

    function_pass_manager.doInitialization();
    for (auto func = llvm_module->begin(); func != llvm_module->end(); ++func)
        function_pass_manager.run(*func);
    function_pass_manager.doFinalization();
    module_pass_manager.run(*llvm_module);

    std::string obj(outstr.begin(), outstr.end());
    std::string obj_file = filename + ".obj";
    std::string gcn_file = filename + ".gcn";
    store_file(obj_file, obj);
    std::string lld_cmd = "ld.lld -shared " + obj_file + " -o " + gcn_file;
    if (std::system(lld_cmd.c_str()))
        error("Generating gcn using lld");

    return load_file(gcn_file);
}
