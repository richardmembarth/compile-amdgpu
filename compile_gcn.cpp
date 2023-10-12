#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <sstream>

#include <lld/Common/Driver.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>

void store_file(const std::string& filename, const std::string& str);
std::string load_file(const std::string& filename);
std::string emit_gcn(const std::string& program, const std::string& cpu, const std::string& filename, llvm::OptimizationLevel opt_level);

int main(int argc, char** argv) {
    std::string filename;
    if (argc == 2)
        filename = argv[1];
    else
        error("usage: % 'llvmir.amdgpu'", argv[0]);

    const std::string& program = load_file(filename);
    emit_gcn(program, "gfx906", filename, llvm::OptimizationLevel::O3);

    return EXIT_SUCCESS;
}

inline std::string read_stream(std::istream& stream) {
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

void store_file(const std::string& filename, const std::string& str) {
    const std::byte* data = reinterpret_cast<const std::byte*>(str.data());
    std::ofstream dst_file(filename, std::ofstream::binary);
    if (!dst_file)
        error("Can't open destination file '%'", filename);
    dst_file.write(reinterpret_cast<const char*>(data), str.length());
}

std::string load_file(const std::string& filename) {
    std::ifstream src_file(filename);
    if (!src_file)
        error("Can't open source file '%'", filename);
    return read_stream(src_file);
}

bool llvm_initialized = false;
std::string emit_gcn(const std::string& program, const std::string& cpu, const std::string& filename, llvm::OptimizationLevel opt_level) {
    if (!llvm_initialized) {
        std::vector<const char*> c_llvm_args;
        std::vector<std::string> llvm_args = { "gcn", "-opt-bisect-limit=-1" };
        for (auto &str : llvm_args)
            c_llvm_args.push_back(str.c_str());
        llvm::cl::ParseCommandLineOptions(c_llvm_args.size(), c_llvm_args.data(), "AnyDSL gcn JIT compiler\n");

        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUAsmParser();
        LLVMInitializeAMDGPUAsmPrinter();
        llvm_initialized = true;
    }

    llvm::LLVMContext llvm_context;
    llvm::SMDiagnostic diagnostic_err;
    std::unique_ptr<llvm::Module> llvm_module = llvm::parseIR(llvm::MemoryBuffer::getMemBuffer(program)->getMemBufferRef(), diagnostic_err, llvm_context);

    auto get_diag_msg = [&] () -> std::string {
        std::string stream;
        llvm::raw_string_ostream llvm_stream(stream);
        diagnostic_err.print("", llvm_stream);
        llvm_stream.flush();
        return stream;
    };

    if (!llvm_module)
        error("Parsing IR file %:\n%", filename, get_diag_msg());

    auto triple_str = llvm_module->getTargetTriple();
    std::string error_str;
    auto target = llvm::TargetRegistry::lookupTarget(triple_str, error_str);
    llvm::TargetOptions options;
    options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    options.NoTrappingFPMath = true;
    std::string attrs = "-trap-handler";
    llvm::TargetMachine* machine = target->createTargetMachine(triple_str, cpu, attrs, options, llvm::Reloc::PIC_, llvm::CodeModel::Small, llvm::CodeGenOpt::Aggressive);

    // link ocml.amdgcn and ocml config
    if (cpu.compare(0, 3, "gfx"))
        error("Expected gfx ISA, got %", cpu);
    std::string isa_version = std::string(&cpu[3]);
    std::string wavefrontsize64 = std::stoi(isa_version) >= 1000 ? "0" : "1";
    std::string bitcode_path(HSA_BITCODE_PATH + std::string("/"));
    std::string bitcode_suffix(HSA_BITCODE_SUFFIX);
    std::string  isa_file = bitcode_path + "oclc_isa_version_" + isa_version + bitcode_suffix;
    std::string ocml_file = bitcode_path + "ocml" + bitcode_suffix;
    std::string ockl_file = bitcode_path + "ockl" + bitcode_suffix;
    std::string ocml_config = R"(; Module anydsl ocml config
                                @__oclc_finite_only_opt = addrspace(4) constant i8 0
                                @__oclc_unsafe_math_opt = addrspace(4) constant i8 0
                                @__oclc_daz_opt = addrspace(4) constant i8 0
                                @__oclc_correctly_rounded_sqrt32 = addrspace(4) constant i8 0
                                @__oclc_wavefrontsize64 = addrspace(4) constant i8 )" + wavefrontsize64;
    std::unique_ptr<llvm::Module> isa_module(llvm::parseIRFile(isa_file, diagnostic_err, llvm_context));
    if (!isa_module)
        error("Can't create isa module for '%':\n%", isa_file, get_diag_msg());
    std::unique_ptr<llvm::Module> config_module = llvm::parseIR(llvm::MemoryBuffer::getMemBuffer(ocml_config)->getMemBufferRef(), diagnostic_err, llvm_context);
    if (!config_module)
        error("Can't create ocml config module:\n%", get_diag_msg());
    std::unique_ptr<llvm::Module> ocml_module(llvm::parseIRFile(ocml_file, diagnostic_err, llvm_context));
    if (!ocml_module)
        error("Can't create ocml module for '%':\n%", ocml_file, get_diag_msg());
    std::unique_ptr<llvm::Module> ockl_module(llvm::parseIRFile(ockl_file, diagnostic_err, llvm_context));
    if (!ockl_module)
        error("Can't create ockl module for '%':\n%", ockl_file, get_diag_msg());

    // override data layout with the one coming from the target machine
    llvm_module->setDataLayout(machine->createDataLayout());
     isa_module->setDataLayout(machine->createDataLayout());
    ocml_module->setDataLayout(machine->createDataLayout());
    ockl_module->setDataLayout(machine->createDataLayout());
    config_module->setDataLayout(machine->createDataLayout());

    llvm::Linker linker(*llvm_module.get());
    if (linker.linkInModule(std::move(ocml_module), llvm::Linker::Flags::LinkOnlyNeeded))
        error("Can't link ocml into module");
    if (linker.linkInModule(std::move(ockl_module), llvm::Linker::Flags::LinkOnlyNeeded))
        error("Can't link ockl into module");
    if (linker.linkInModule(std::move(isa_module), llvm::Linker::Flags::None))
        error("Can't link isa into module");
    if (linker.linkInModule(std::move(config_module), llvm::Linker::Flags::None))
        error("Can't link config into module");

    auto run_pass_manager = [&] (std::unique_ptr<llvm::Module> module, llvm::CodeGenFileType cogen_file_type, std::string out_filename, bool print_ir=false) {
        machine->Options.MCOptions.AsmVerbose = true;

        // create the analysis managers
        llvm::LoopAnalysisManager LAM;
        llvm::FunctionAnalysisManager FAM;
        llvm::CGSCCAnalysisManager CGAM;
        llvm::ModuleAnalysisManager MAM;

        llvm::PassBuilder PB(machine);

        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

        llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(opt_level);

        MPM.run(*module, MAM);

        llvm::legacy::PassManager module_pass_manager;
        llvm::SmallString<0> outstr;
        llvm::raw_svector_ostream llvm_stream(outstr);
        machine->addPassesToEmitFile(module_pass_manager, llvm_stream, nullptr, cogen_file_type, true);
        module_pass_manager.run(*module);

        if (print_ir) {
            std::error_code EC;
            llvm::raw_fd_ostream outstream(filename + "_final.ll", EC);
            module->print(outstream, nullptr);
        }

        std::string out(outstr.begin(), outstr.end());
        store_file(out_filename, out);
    };

    std::string asm_file = filename + ".asm";
    std::string obj_file = filename + ".obj";
    std::string gcn_file = filename + ".gcn";

    bool print_ir = false;
    if (print_ir)
        run_pass_manager(llvm::CloneModule(*llvm_module.get()), llvm::CodeGenFileType::CGFT_AssemblyFile, asm_file, print_ir);
    run_pass_manager(std::move(llvm_module), llvm::CodeGenFileType::CGFT_ObjectFile, obj_file);

    llvm::raw_os_ostream lld_cout(std::cout);
    llvm::raw_os_ostream lld_cerr(std::cerr);
    std::vector<const char*> lld_args = {
        "ld",
        "-shared",
        obj_file.c_str(),
        "-o",
        gcn_file.c_str()
    };
    if (!lld::elf::link(lld_args, lld_cout, lld_cerr, false, false))
        error("Generating gcn using ld");

    return load_file(gcn_file);
}
