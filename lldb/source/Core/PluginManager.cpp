//===-- PluginManager.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/PluginManager.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Symbol/SaveCoreOptions.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StringList.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#if defined(_WIN32)
#include "lldb/Host/windows/PosixApi.h"
#endif

using namespace lldb;
using namespace lldb_private;

typedef bool (*PluginInitCallback)();
typedef void (*PluginTermCallback)();

struct PluginInfo {
  PluginInfo() = default;

  llvm::sys::DynamicLibrary library;
  PluginInitCallback plugin_init_callback = nullptr;
  PluginTermCallback plugin_term_callback = nullptr;
};

typedef std::map<FileSpec, PluginInfo> PluginTerminateMap;

static std::recursive_mutex &GetPluginMapMutex() {
  static std::recursive_mutex g_plugin_map_mutex;
  return g_plugin_map_mutex;
}

static PluginTerminateMap &GetPluginMap() {
  static PluginTerminateMap g_plugin_map;
  return g_plugin_map;
}

static bool PluginIsLoaded(const FileSpec &plugin_file_spec) {
  std::lock_guard<std::recursive_mutex> guard(GetPluginMapMutex());
  PluginTerminateMap &plugin_map = GetPluginMap();
  return plugin_map.find(plugin_file_spec) != plugin_map.end();
}

static void SetPluginInfo(const FileSpec &plugin_file_spec,
                          const PluginInfo &plugin_info) {
  std::lock_guard<std::recursive_mutex> guard(GetPluginMapMutex());
  PluginTerminateMap &plugin_map = GetPluginMap();
  assert(plugin_map.find(plugin_file_spec) == plugin_map.end());
  plugin_map[plugin_file_spec] = plugin_info;
}

template <typename FPtrTy> static FPtrTy CastToFPtr(void *VPtr) {
  return reinterpret_cast<FPtrTy>(VPtr);
}

static FileSystem::EnumerateDirectoryResult
LoadPluginCallback(void *baton, llvm::sys::fs::file_type ft,
                   llvm::StringRef path) {
  Status error;

  namespace fs = llvm::sys::fs;
  // If we have a regular file, a symbolic link or unknown file type, try and
  // process the file. We must handle unknown as sometimes the directory
  // enumeration might be enumerating a file system that doesn't have correct
  // file type information.
  if (ft == fs::file_type::regular_file || ft == fs::file_type::symlink_file ||
      ft == fs::file_type::type_unknown) {
    FileSpec plugin_file_spec(path);
    FileSystem::Instance().Resolve(plugin_file_spec);

    if (PluginIsLoaded(plugin_file_spec))
      return FileSystem::eEnumerateDirectoryResultNext;
    else {
      PluginInfo plugin_info;

      std::string pluginLoadError;
      plugin_info.library = llvm::sys::DynamicLibrary::getPermanentLibrary(
          plugin_file_spec.GetPath().c_str(), &pluginLoadError);
      if (plugin_info.library.isValid()) {
        bool success = false;
        plugin_info.plugin_init_callback = CastToFPtr<PluginInitCallback>(
            plugin_info.library.getAddressOfSymbol("LLDBPluginInitialize"));
        if (plugin_info.plugin_init_callback) {
          // Call the plug-in "bool LLDBPluginInitialize(void)" function
          success = plugin_info.plugin_init_callback();
        }

        if (success) {
          // It is ok for the "LLDBPluginTerminate" symbol to be nullptr
          plugin_info.plugin_term_callback = CastToFPtr<PluginTermCallback>(
              plugin_info.library.getAddressOfSymbol("LLDBPluginTerminate"));
        } else {
          // The initialize function returned FALSE which means the plug-in
          // might not be compatible, or might be too new or too old, or might
          // not want to run on this machine.  Set it to a default-constructed
          // instance to invalidate it.
          plugin_info = PluginInfo();
        }

        // Regardless of success or failure, cache the plug-in load in our
        // plug-in info so we don't try to load it again and again.
        SetPluginInfo(plugin_file_spec, plugin_info);

        return FileSystem::eEnumerateDirectoryResultNext;
      }
    }
  }

  if (ft == fs::file_type::directory_file ||
      ft == fs::file_type::symlink_file || ft == fs::file_type::type_unknown) {
    // Try and recurse into anything that a directory or symbolic link. We must
    // also do this for unknown as sometimes the directory enumeration might be
    // enumerating a file system that doesn't have correct file type
    // information.
    return FileSystem::eEnumerateDirectoryResultEnter;
  }

  return FileSystem::eEnumerateDirectoryResultNext;
}

void PluginManager::Initialize() {
  const bool find_directories = true;
  const bool find_files = true;
  const bool find_other = true;
  char dir_path[PATH_MAX];
  if (FileSpec dir_spec = HostInfo::GetSystemPluginDir()) {
    if (FileSystem::Instance().Exists(dir_spec) &&
        dir_spec.GetPath(dir_path, sizeof(dir_path))) {
      FileSystem::Instance().EnumerateDirectory(dir_path, find_directories,
                                                find_files, find_other,
                                                LoadPluginCallback, nullptr);
    }
  }

  if (FileSpec dir_spec = HostInfo::GetUserPluginDir()) {
    if (FileSystem::Instance().Exists(dir_spec) &&
        dir_spec.GetPath(dir_path, sizeof(dir_path))) {
      FileSystem::Instance().EnumerateDirectory(dir_path, find_directories,
                                                find_files, find_other,
                                                LoadPluginCallback, nullptr);
    }
  }
}

void PluginManager::Terminate() {
  std::lock_guard<std::recursive_mutex> guard(GetPluginMapMutex());
  PluginTerminateMap &plugin_map = GetPluginMap();

  PluginTerminateMap::const_iterator pos, end = plugin_map.end();
  for (pos = plugin_map.begin(); pos != end; ++pos) {
    // Call the plug-in "void LLDBPluginTerminate (void)" function if there is
    // one (if the symbol was not nullptr).
    if (pos->second.library.isValid()) {
      if (pos->second.plugin_term_callback)
        pos->second.plugin_term_callback();
    }
  }
  plugin_map.clear();
}

llvm::ArrayRef<PluginNamespace> PluginManager::GetPluginNamespaces() {
  static PluginNamespace PluginNamespaces[] = {

      {
          "abi",
          PluginManager::GetABIPluginInfo,
          PluginManager::SetABIPluginEnabled,
      },

      {
          "architecture",
          PluginManager::GetArchitecturePluginInfo,
          PluginManager::SetArchitecturePluginEnabled,
      },

      {
          "disassembler",
          PluginManager::GetDisassemblerPluginInfo,
          PluginManager::SetDisassemblerPluginEnabled,
      },

      {
          "dynamic-loader",
          PluginManager::GetDynamicLoaderPluginInfo,
          PluginManager::SetDynamicLoaderPluginEnabled,
      },

      {
          "emulate-instruction",
          PluginManager::GetEmulateInstructionPluginInfo,
          PluginManager::SetEmulateInstructionPluginEnabled,
      },

      {
          "instrumentation-runtime",
          PluginManager::GetInstrumentationRuntimePluginInfo,
          PluginManager::SetInstrumentationRuntimePluginEnabled,
      },

      {
          "jit-loader",
          PluginManager::GetJITLoaderPluginInfo,
          PluginManager::SetJITLoaderPluginEnabled,
      },

      {
          "language",
          PluginManager::GetLanguagePluginInfo,
          PluginManager::SetLanguagePluginEnabled,
      },

      {
          "language-runtime",
          PluginManager::GetLanguageRuntimePluginInfo,
          PluginManager::SetLanguageRuntimePluginEnabled,
      },

      {
          "memory-history",
          PluginManager::GetMemoryHistoryPluginInfo,
          PluginManager::SetMemoryHistoryPluginEnabled,
      },

      {
          "object-container",
          PluginManager::GetObjectContainerPluginInfo,
          PluginManager::SetObjectContainerPluginEnabled,
      },

      {
          "object-file",
          PluginManager::GetObjectFilePluginInfo,
          PluginManager::SetObjectFilePluginEnabled,
      },

      {
          "operating-system",
          PluginManager::GetOperatingSystemPluginInfo,
          PluginManager::SetOperatingSystemPluginEnabled,
      },

      {
          "platform",
          PluginManager::GetPlatformPluginInfo,
          PluginManager::SetPlatformPluginEnabled,
      },

      {
          "process",
          PluginManager::GetProcessPluginInfo,
          PluginManager::SetProcessPluginEnabled,
      },

      {
          "repl",
          PluginManager::GetREPLPluginInfo,
          PluginManager::SetREPLPluginEnabled,
      },

      {
          "register-type-builder",
          PluginManager::GetRegisterTypeBuilderPluginInfo,
          PluginManager::SetRegisterTypeBuilderPluginEnabled,
      },

      {
          "script-interpreter",
          PluginManager::GetScriptInterpreterPluginInfo,
          PluginManager::SetScriptInterpreterPluginEnabled,
      },

      {
          "scripted-interface",
          PluginManager::GetScriptedInterfacePluginInfo,
          PluginManager::SetScriptedInterfacePluginEnabled,
      },

      {
          "structured-data",
          PluginManager::GetStructuredDataPluginInfo,
          PluginManager::SetStructuredDataPluginEnabled,
      },

      {
          "symbol-file",
          PluginManager::GetSymbolFilePluginInfo,
          PluginManager::SetSymbolFilePluginEnabled,
      },

      {
          "symbol-locator",
          PluginManager::GetSymbolLocatorPluginInfo,
          PluginManager::SetSymbolLocatorPluginEnabled,
      },

      {
          "symbol-vendor",
          PluginManager::GetSymbolVendorPluginInfo,
          PluginManager::SetSymbolVendorPluginEnabled,
      },

      {
          "system-runtime",
          PluginManager::GetSystemRuntimePluginInfo,
          PluginManager::SetSystemRuntimePluginEnabled,
      },

      {
          "trace",
          PluginManager::GetTracePluginInfo,
          PluginManager::SetTracePluginEnabled,
      },

      {
          "trace-exporter",
          PluginManager::GetTraceExporterPluginInfo,
          PluginManager::SetTraceExporterPluginEnabled,
      },

      {
          "type-system",
          PluginManager::GetTypeSystemPluginInfo,
          PluginManager::SetTypeSystemPluginEnabled,
      },

      {
          "unwind-assembly",
          PluginManager::GetUnwindAssemblyPluginInfo,
          PluginManager::SetUnwindAssemblyPluginEnabled,
      },
  };

  return PluginNamespaces;
}

llvm::json::Object PluginManager::GetJSON(llvm::StringRef pattern) {
  llvm::json::Object plugin_stats;

  for (const PluginNamespace &plugin_ns : GetPluginNamespaces()) {
    llvm::json::Array namespace_stats;

    for (const RegisteredPluginInfo &plugin : plugin_ns.get_info()) {
      if (MatchPluginName(pattern, plugin_ns, plugin)) {
        llvm::json::Object plugin_json;
        plugin_json.try_emplace("name", plugin.name);
        plugin_json.try_emplace("enabled", plugin.enabled);
        namespace_stats.emplace_back(std::move(plugin_json));
      }
    }
    if (!namespace_stats.empty())
      plugin_stats.try_emplace(plugin_ns.name, std::move(namespace_stats));
  }

  return plugin_stats;
}

bool PluginManager::MatchPluginName(llvm::StringRef pattern,
                                    const PluginNamespace &plugin_ns,
                                    const RegisteredPluginInfo &plugin_info) {
  // The empty pattern matches all plugins.
  if (pattern.empty())
    return true;

  // Check if the pattern matches the namespace.
  if (pattern == plugin_ns.name)
    return true;

  // Check if the pattern matches the qualified name.
  std::string qualified_name = (plugin_ns.name + "." + plugin_info.name).str();
  return pattern == qualified_name;
}

template <typename Callback> struct PluginInstance {
  typedef Callback CallbackType;

  PluginInstance() = default;
  PluginInstance(llvm::StringRef name, llvm::StringRef description,
                 Callback create_callback,
                 DebuggerInitializeCallback debugger_init_callback = nullptr)
      : name(name), description(description), enabled(true),
        create_callback(create_callback),
        debugger_init_callback(debugger_init_callback) {}

  llvm::StringRef name;
  llvm::StringRef description;
  bool enabled;
  Callback create_callback;
  DebuggerInitializeCallback debugger_init_callback;
};

template <typename Instance> class PluginInstances {
public:
  template <typename... Args>
  bool RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                      typename Instance::CallbackType callback,
                      Args &&...args) {
    if (!callback)
      return false;
    assert(!name.empty());
    m_instances.emplace_back(name, description, callback,
                             std::forward<Args>(args)...);
    return true;
  }

  bool UnregisterPlugin(typename Instance::CallbackType callback) {
    if (!callback)
      return false;
    auto pos = m_instances.begin();
    auto end = m_instances.end();
    for (; pos != end; ++pos) {
      if (pos->create_callback == callback) {
        m_instances.erase(pos);
        return true;
      }
    }
    return false;
  }

  typename Instance::CallbackType GetCallbackAtIndex(uint32_t idx) {
    if (const Instance *instance = GetInstanceAtIndex(idx))
      return instance->create_callback;
    return nullptr;
  }

  llvm::StringRef GetDescriptionAtIndex(uint32_t idx) {
    if (const Instance *instance = GetInstanceAtIndex(idx))
      return instance->description;
    return "";
  }

  llvm::StringRef GetNameAtIndex(uint32_t idx) {
    if (const Instance *instance = GetInstanceAtIndex(idx))
      return instance->name;
    return "";
  }

  typename Instance::CallbackType GetCallbackForName(llvm::StringRef name) {
    if (const Instance *instance = GetInstanceForName(name))
      return instance->create_callback;
    return nullptr;
  }

  void PerformDebuggerCallback(Debugger &debugger) {
    for (const auto &instance : m_instances) {
      if (!instance.enabled)
        continue;
      if (instance.debugger_init_callback)
        instance.debugger_init_callback(debugger);
    }
  }

  // Return a copy of all the enabled instances.
  // Note that this is a copy of the internal state so modifications
  // to the returned instances will not be reflected back to instances
  // stored by the PluginInstances object.
  std::vector<Instance> GetSnapshot() {
    std::vector<Instance> enabled_instances;
    for (const auto &instance : m_instances) {
      if (instance.enabled)
        enabled_instances.push_back(instance);
    }
    return enabled_instances;
  }

  const Instance *GetInstanceAtIndex(uint32_t idx) {
    uint32_t count = 0;

    return FindEnabledInstance(
        [&](const Instance &instance) { return count++ == idx; });
  }

  const Instance *GetInstanceForName(llvm::StringRef name) {
    if (name.empty())
      return nullptr;

    return FindEnabledInstance(
        [&](const Instance &instance) { return instance.name == name; });
  }

  const Instance *
  FindEnabledInstance(std::function<bool(const Instance &)> predicate) const {
    for (const auto &instance : m_instances) {
      if (!instance.enabled)
        continue;
      if (predicate(instance))
        return &instance;
    }
    return nullptr;
  }

  // Return a list of all the registered plugin instances. This includes both
  // enabled and disabled instances. The instances are listed in the order they
  // were registered which is the order they would be queried if they were all
  // enabled.
  std::vector<RegisteredPluginInfo> GetPluginInfoForAllInstances() {
    // Lookup the plugin info for each instance in the sorted order.
    std::vector<RegisteredPluginInfo> plugin_infos;
    plugin_infos.reserve(m_instances.size());
    for (const Instance &instance : m_instances)
      plugin_infos.push_back(
          {instance.name, instance.description, instance.enabled});

    return plugin_infos;
  }

  bool SetInstanceEnabled(llvm::StringRef name, bool enable) {
    auto it = llvm::find_if(m_instances, [&](const Instance &instance) {
      return instance.name == name;
    });

    if (it == m_instances.end())
      return false;

    it->enabled = enable;
    return true;
  }

private:
  std::vector<Instance> m_instances;
};

#pragma mark ABI

typedef PluginInstance<ABICreateInstance> ABIInstance;
typedef PluginInstances<ABIInstance> ABIInstances;

static ABIInstances &GetABIInstances() {
  static ABIInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(llvm::StringRef name,
                                   llvm::StringRef description,
                                   ABICreateInstance create_callback) {
  return GetABIInstances().RegisterPlugin(name, description, create_callback);
}

bool PluginManager::UnregisterPlugin(ABICreateInstance create_callback) {
  return GetABIInstances().UnregisterPlugin(create_callback);
}

ABICreateInstance PluginManager::GetABICreateCallbackAtIndex(uint32_t idx) {
  return GetABIInstances().GetCallbackAtIndex(idx);
}

#pragma mark Architecture

typedef PluginInstance<ArchitectureCreateInstance> ArchitectureInstance;
typedef PluginInstances<ArchitectureInstance> ArchitectureInstances;

static ArchitectureInstances &GetArchitectureInstances() {
  static ArchitectureInstances g_instances;
  return g_instances;
}

void PluginManager::RegisterPlugin(llvm::StringRef name,
                                   llvm::StringRef description,
                                   ArchitectureCreateInstance create_callback) {
  GetArchitectureInstances().RegisterPlugin(name, description, create_callback);
}

void PluginManager::UnregisterPlugin(
    ArchitectureCreateInstance create_callback) {
  auto &instances = GetArchitectureInstances();
  instances.UnregisterPlugin(create_callback);
}

std::unique_ptr<Architecture>
PluginManager::CreateArchitectureInstance(const ArchSpec &arch) {
  for (const auto &instances : GetArchitectureInstances().GetSnapshot()) {
    if (auto plugin_up = instances.create_callback(arch))
      return plugin_up;
  }
  return nullptr;
}

#pragma mark Disassembler

typedef PluginInstance<DisassemblerCreateInstance> DisassemblerInstance;
typedef PluginInstances<DisassemblerInstance> DisassemblerInstances;

static DisassemblerInstances &GetDisassemblerInstances() {
  static DisassemblerInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(llvm::StringRef name,
                                   llvm::StringRef description,
                                   DisassemblerCreateInstance create_callback) {
  return GetDisassemblerInstances().RegisterPlugin(name, description,
                                                   create_callback);
}

bool PluginManager::UnregisterPlugin(
    DisassemblerCreateInstance create_callback) {
  return GetDisassemblerInstances().UnregisterPlugin(create_callback);
}

DisassemblerCreateInstance
PluginManager::GetDisassemblerCreateCallbackAtIndex(uint32_t idx) {
  return GetDisassemblerInstances().GetCallbackAtIndex(idx);
}

DisassemblerCreateInstance
PluginManager::GetDisassemblerCreateCallbackForPluginName(
    llvm::StringRef name) {
  return GetDisassemblerInstances().GetCallbackForName(name);
}

#pragma mark DynamicLoader

typedef PluginInstance<DynamicLoaderCreateInstance> DynamicLoaderInstance;
typedef PluginInstances<DynamicLoaderInstance> DynamicLoaderInstances;

static DynamicLoaderInstances &GetDynamicLoaderInstances() {
  static DynamicLoaderInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    DynamicLoaderCreateInstance create_callback,
    DebuggerInitializeCallback debugger_init_callback) {
  return GetDynamicLoaderInstances().RegisterPlugin(
      name, description, create_callback, debugger_init_callback);
}

bool PluginManager::UnregisterPlugin(
    DynamicLoaderCreateInstance create_callback) {
  return GetDynamicLoaderInstances().UnregisterPlugin(create_callback);
}

DynamicLoaderCreateInstance
PluginManager::GetDynamicLoaderCreateCallbackAtIndex(uint32_t idx) {
  return GetDynamicLoaderInstances().GetCallbackAtIndex(idx);
}

DynamicLoaderCreateInstance
PluginManager::GetDynamicLoaderCreateCallbackForPluginName(
    llvm::StringRef name) {
  return GetDynamicLoaderInstances().GetCallbackForName(name);
}

#pragma mark JITLoader

typedef PluginInstance<JITLoaderCreateInstance> JITLoaderInstance;
typedef PluginInstances<JITLoaderInstance> JITLoaderInstances;

static JITLoaderInstances &GetJITLoaderInstances() {
  static JITLoaderInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    JITLoaderCreateInstance create_callback,
    DebuggerInitializeCallback debugger_init_callback) {
  return GetJITLoaderInstances().RegisterPlugin(
      name, description, create_callback, debugger_init_callback);
}

bool PluginManager::UnregisterPlugin(JITLoaderCreateInstance create_callback) {
  return GetJITLoaderInstances().UnregisterPlugin(create_callback);
}

JITLoaderCreateInstance
PluginManager::GetJITLoaderCreateCallbackAtIndex(uint32_t idx) {
  return GetJITLoaderInstances().GetCallbackAtIndex(idx);
}

#pragma mark EmulateInstruction

typedef PluginInstance<EmulateInstructionCreateInstance>
    EmulateInstructionInstance;
typedef PluginInstances<EmulateInstructionInstance> EmulateInstructionInstances;

static EmulateInstructionInstances &GetEmulateInstructionInstances() {
  static EmulateInstructionInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    EmulateInstructionCreateInstance create_callback) {
  return GetEmulateInstructionInstances().RegisterPlugin(name, description,
                                                         create_callback);
}

bool PluginManager::UnregisterPlugin(
    EmulateInstructionCreateInstance create_callback) {
  return GetEmulateInstructionInstances().UnregisterPlugin(create_callback);
}

EmulateInstructionCreateInstance
PluginManager::GetEmulateInstructionCreateCallbackAtIndex(uint32_t idx) {
  return GetEmulateInstructionInstances().GetCallbackAtIndex(idx);
}

EmulateInstructionCreateInstance
PluginManager::GetEmulateInstructionCreateCallbackForPluginName(
    llvm::StringRef name) {
  return GetEmulateInstructionInstances().GetCallbackForName(name);
}

#pragma mark OperatingSystem

typedef PluginInstance<OperatingSystemCreateInstance> OperatingSystemInstance;
typedef PluginInstances<OperatingSystemInstance> OperatingSystemInstances;

static OperatingSystemInstances &GetOperatingSystemInstances() {
  static OperatingSystemInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    OperatingSystemCreateInstance create_callback,
    DebuggerInitializeCallback debugger_init_callback) {
  return GetOperatingSystemInstances().RegisterPlugin(
      name, description, create_callback, debugger_init_callback);
}

bool PluginManager::UnregisterPlugin(
    OperatingSystemCreateInstance create_callback) {
  return GetOperatingSystemInstances().UnregisterPlugin(create_callback);
}

OperatingSystemCreateInstance
PluginManager::GetOperatingSystemCreateCallbackAtIndex(uint32_t idx) {
  return GetOperatingSystemInstances().GetCallbackAtIndex(idx);
}

OperatingSystemCreateInstance
PluginManager::GetOperatingSystemCreateCallbackForPluginName(
    llvm::StringRef name) {
  return GetOperatingSystemInstances().GetCallbackForName(name);
}

#pragma mark Language

typedef PluginInstance<LanguageCreateInstance> LanguageInstance;
typedef PluginInstances<LanguageInstance> LanguageInstances;

static LanguageInstances &GetLanguageInstances() {
  static LanguageInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    LanguageCreateInstance create_callback,
    DebuggerInitializeCallback debugger_init_callback) {
  return GetLanguageInstances().RegisterPlugin(
      name, description, create_callback, debugger_init_callback);
}

bool PluginManager::UnregisterPlugin(LanguageCreateInstance create_callback) {
  return GetLanguageInstances().UnregisterPlugin(create_callback);
}

LanguageCreateInstance
PluginManager::GetLanguageCreateCallbackAtIndex(uint32_t idx) {
  return GetLanguageInstances().GetCallbackAtIndex(idx);
}

#pragma mark LanguageRuntime

struct LanguageRuntimeInstance
    : public PluginInstance<LanguageRuntimeCreateInstance> {
  LanguageRuntimeInstance(
      llvm::StringRef name, llvm::StringRef description,
      CallbackType create_callback,
      DebuggerInitializeCallback debugger_init_callback,
      LanguageRuntimeGetCommandObject command_callback,
      LanguageRuntimeGetExceptionPrecondition precondition_callback)
      : PluginInstance<LanguageRuntimeCreateInstance>(
            name, description, create_callback, debugger_init_callback),
        command_callback(command_callback),
        precondition_callback(precondition_callback) {}

  LanguageRuntimeGetCommandObject command_callback;
  LanguageRuntimeGetExceptionPrecondition precondition_callback;
};

typedef PluginInstances<LanguageRuntimeInstance> LanguageRuntimeInstances;

static LanguageRuntimeInstances &GetLanguageRuntimeInstances() {
  static LanguageRuntimeInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    LanguageRuntimeCreateInstance create_callback,
    LanguageRuntimeGetCommandObject command_callback,
    LanguageRuntimeGetExceptionPrecondition precondition_callback) {
  return GetLanguageRuntimeInstances().RegisterPlugin(
      name, description, create_callback, nullptr, command_callback,
      precondition_callback);
}

bool PluginManager::UnregisterPlugin(
    LanguageRuntimeCreateInstance create_callback) {
  return GetLanguageRuntimeInstances().UnregisterPlugin(create_callback);
}

LanguageRuntimeCreateInstance
PluginManager::GetLanguageRuntimeCreateCallbackAtIndex(uint32_t idx) {
  return GetLanguageRuntimeInstances().GetCallbackAtIndex(idx);
}

LanguageRuntimeGetCommandObject
PluginManager::GetLanguageRuntimeGetCommandObjectAtIndex(uint32_t idx) {
  if (auto instance = GetLanguageRuntimeInstances().GetInstanceAtIndex(idx))
    return instance->command_callback;
  return nullptr;
}

LanguageRuntimeGetExceptionPrecondition
PluginManager::GetLanguageRuntimeGetExceptionPreconditionAtIndex(uint32_t idx) {
  if (auto instance = GetLanguageRuntimeInstances().GetInstanceAtIndex(idx))
    return instance->precondition_callback;
  return nullptr;
}

#pragma mark SystemRuntime

typedef PluginInstance<SystemRuntimeCreateInstance> SystemRuntimeInstance;
typedef PluginInstances<SystemRuntimeInstance> SystemRuntimeInstances;

static SystemRuntimeInstances &GetSystemRuntimeInstances() {
  static SystemRuntimeInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    SystemRuntimeCreateInstance create_callback) {
  return GetSystemRuntimeInstances().RegisterPlugin(name, description,
                                                    create_callback);
}

bool PluginManager::UnregisterPlugin(
    SystemRuntimeCreateInstance create_callback) {
  return GetSystemRuntimeInstances().UnregisterPlugin(create_callback);
}

SystemRuntimeCreateInstance
PluginManager::GetSystemRuntimeCreateCallbackAtIndex(uint32_t idx) {
  return GetSystemRuntimeInstances().GetCallbackAtIndex(idx);
}

#pragma mark ObjectFile

struct ObjectFileInstance : public PluginInstance<ObjectFileCreateInstance> {
  ObjectFileInstance(
      llvm::StringRef name, llvm::StringRef description,
      CallbackType create_callback,
      ObjectFileCreateMemoryInstance create_memory_callback,
      ObjectFileGetModuleSpecifications get_module_specifications,
      ObjectFileSaveCore save_core,
      DebuggerInitializeCallback debugger_init_callback)
      : PluginInstance<ObjectFileCreateInstance>(
            name, description, create_callback, debugger_init_callback),
        create_memory_callback(create_memory_callback),
        get_module_specifications(get_module_specifications),
        save_core(save_core) {}

  ObjectFileCreateMemoryInstance create_memory_callback;
  ObjectFileGetModuleSpecifications get_module_specifications;
  ObjectFileSaveCore save_core;
};
typedef PluginInstances<ObjectFileInstance> ObjectFileInstances;

static ObjectFileInstances &GetObjectFileInstances() {
  static ObjectFileInstances g_instances;
  return g_instances;
}

bool PluginManager::IsRegisteredObjectFilePluginName(llvm::StringRef name) {
  if (name.empty())
    return false;

  return GetObjectFileInstances().GetInstanceForName(name) != nullptr;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    ObjectFileCreateInstance create_callback,
    ObjectFileCreateMemoryInstance create_memory_callback,
    ObjectFileGetModuleSpecifications get_module_specifications,
    ObjectFileSaveCore save_core,
    DebuggerInitializeCallback debugger_init_callback) {
  return GetObjectFileInstances().RegisterPlugin(
      name, description, create_callback, create_memory_callback,
      get_module_specifications, save_core, debugger_init_callback);
}

bool PluginManager::UnregisterPlugin(ObjectFileCreateInstance create_callback) {
  return GetObjectFileInstances().UnregisterPlugin(create_callback);
}

ObjectFileCreateInstance
PluginManager::GetObjectFileCreateCallbackAtIndex(uint32_t idx) {
  return GetObjectFileInstances().GetCallbackAtIndex(idx);
}

ObjectFileCreateMemoryInstance
PluginManager::GetObjectFileCreateMemoryCallbackAtIndex(uint32_t idx) {
  if (auto instance = GetObjectFileInstances().GetInstanceAtIndex(idx))
    return instance->create_memory_callback;
  return nullptr;
}

ObjectFileGetModuleSpecifications
PluginManager::GetObjectFileGetModuleSpecificationsCallbackAtIndex(
    uint32_t idx) {
  if (auto instance = GetObjectFileInstances().GetInstanceAtIndex(idx))
    return instance->get_module_specifications;
  return nullptr;
}

ObjectFileCreateMemoryInstance
PluginManager::GetObjectFileCreateMemoryCallbackForPluginName(
    llvm::StringRef name) {
  if (auto instance = GetObjectFileInstances().GetInstanceForName(name))
    return instance->create_memory_callback;
  return nullptr;
}

Status PluginManager::SaveCore(lldb_private::SaveCoreOptions &options) {
  Status error;
  if (!options.GetOutputFile()) {
    error = Status::FromErrorString("No output file specified");
    return error;
  }

  if (!options.GetProcess()) {
    error = Status::FromErrorString("Invalid process");
    return error;
  }

  error = options.EnsureValidConfiguration();
  if (error.Fail())
    return error;

  if (!options.GetPluginName().has_value()) {
    // Try saving core directly from the process plugin first.
    llvm::Expected<bool> ret =
        options.GetProcess()->SaveCore(options.GetOutputFile()->GetPath());
    if (!ret)
      return Status::FromError(ret.takeError());
    if (ret.get())
      return Status();
  }

  // Fall back to object plugins.
  const auto &plugin_name = options.GetPluginName().value_or("");
  auto instances = GetObjectFileInstances().GetSnapshot();
  for (auto &instance : instances) {
    if (plugin_name.empty() || instance.name == plugin_name) {
      // TODO: Refactor the instance.save_core() to not require a process and
      // get it from options instead.
      if (instance.save_core &&
          instance.save_core(options.GetProcess(), options, error))
        return error;
    }
  }

  // Check to see if any of the object file plugins tried and failed to save.
  // if any failure, return the error message.
  if (error.Fail())
    return error;

  // Report only for the plugin that was specified.
  if (!plugin_name.empty())
    return Status::FromErrorStringWithFormatv(
        "The \"{}\" plugin is not able to save a core for this process.",
        plugin_name);

  return Status::FromErrorString(
      "no ObjectFile plugins were able to save a core for this process");
}

std::vector<llvm::StringRef> PluginManager::GetSaveCorePluginNames() {
  std::vector<llvm::StringRef> plugin_names;
  auto instances = GetObjectFileInstances().GetSnapshot();
  for (auto &instance : instances) {
    if (instance.save_core)
      plugin_names.emplace_back(instance.name);
  }
  return plugin_names;
}

#pragma mark ObjectContainer

struct ObjectContainerInstance
    : public PluginInstance<ObjectContainerCreateInstance> {
  ObjectContainerInstance(
      llvm::StringRef name, llvm::StringRef description,
      CallbackType create_callback,
      ObjectContainerCreateMemoryInstance create_memory_callback,
      ObjectFileGetModuleSpecifications get_module_specifications)
      : PluginInstance<ObjectContainerCreateInstance>(name, description,
                                                      create_callback),
        create_memory_callback(create_memory_callback),
        get_module_specifications(get_module_specifications) {}

  ObjectContainerCreateMemoryInstance create_memory_callback;
  ObjectFileGetModuleSpecifications get_module_specifications;
};
typedef PluginInstances<ObjectContainerInstance> ObjectContainerInstances;

static ObjectContainerInstances &GetObjectContainerInstances() {
  static ObjectContainerInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    ObjectContainerCreateInstance create_callback,
    ObjectFileGetModuleSpecifications get_module_specifications,
    ObjectContainerCreateMemoryInstance create_memory_callback) {
  return GetObjectContainerInstances().RegisterPlugin(
      name, description, create_callback, create_memory_callback,
      get_module_specifications);
}

bool PluginManager::UnregisterPlugin(
    ObjectContainerCreateInstance create_callback) {
  return GetObjectContainerInstances().UnregisterPlugin(create_callback);
}

ObjectContainerCreateInstance
PluginManager::GetObjectContainerCreateCallbackAtIndex(uint32_t idx) {
  return GetObjectContainerInstances().GetCallbackAtIndex(idx);
}

ObjectContainerCreateMemoryInstance
PluginManager::GetObjectContainerCreateMemoryCallbackAtIndex(uint32_t idx) {
  if (auto instance = GetObjectContainerInstances().GetInstanceAtIndex(idx))
    return instance->create_memory_callback;
  return nullptr;
}

ObjectFileGetModuleSpecifications
PluginManager::GetObjectContainerGetModuleSpecificationsCallbackAtIndex(
    uint32_t idx) {
  if (auto instance = GetObjectContainerInstances().GetInstanceAtIndex(idx))
    return instance->get_module_specifications;
  return nullptr;
}

#pragma mark Platform

typedef PluginInstance<PlatformCreateInstance> PlatformInstance;
typedef PluginInstances<PlatformInstance> PlatformInstances;

static PlatformInstances &GetPlatformInstances() {
  static PlatformInstances g_platform_instances;
  return g_platform_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    PlatformCreateInstance create_callback,
    DebuggerInitializeCallback debugger_init_callback) {
  return GetPlatformInstances().RegisterPlugin(
      name, description, create_callback, debugger_init_callback);
}

bool PluginManager::UnregisterPlugin(PlatformCreateInstance create_callback) {
  return GetPlatformInstances().UnregisterPlugin(create_callback);
}

llvm::StringRef PluginManager::GetPlatformPluginNameAtIndex(uint32_t idx) {
  return GetPlatformInstances().GetNameAtIndex(idx);
}

llvm::StringRef
PluginManager::GetPlatformPluginDescriptionAtIndex(uint32_t idx) {
  return GetPlatformInstances().GetDescriptionAtIndex(idx);
}

PlatformCreateInstance
PluginManager::GetPlatformCreateCallbackAtIndex(uint32_t idx) {
  return GetPlatformInstances().GetCallbackAtIndex(idx);
}

PlatformCreateInstance
PluginManager::GetPlatformCreateCallbackForPluginName(llvm::StringRef name) {
  return GetPlatformInstances().GetCallbackForName(name);
}

void PluginManager::AutoCompletePlatformName(llvm::StringRef name,
                                             CompletionRequest &request) {
  for (const auto &instance : GetPlatformInstances().GetSnapshot()) {
    if (instance.name.starts_with(name))
      request.AddCompletion(instance.name);
  }
}

#pragma mark Process

typedef PluginInstance<ProcessCreateInstance> ProcessInstance;
typedef PluginInstances<ProcessInstance> ProcessInstances;

static ProcessInstances &GetProcessInstances() {
  static ProcessInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    ProcessCreateInstance create_callback,
    DebuggerInitializeCallback debugger_init_callback) {
  return GetProcessInstances().RegisterPlugin(
      name, description, create_callback, debugger_init_callback);
}

bool PluginManager::UnregisterPlugin(ProcessCreateInstance create_callback) {
  return GetProcessInstances().UnregisterPlugin(create_callback);
}

llvm::StringRef PluginManager::GetProcessPluginNameAtIndex(uint32_t idx) {
  return GetProcessInstances().GetNameAtIndex(idx);
}

llvm::StringRef PluginManager::GetProcessPluginDescriptionAtIndex(uint32_t idx) {
  return GetProcessInstances().GetDescriptionAtIndex(idx);
}

ProcessCreateInstance
PluginManager::GetProcessCreateCallbackAtIndex(uint32_t idx) {
  return GetProcessInstances().GetCallbackAtIndex(idx);
}

ProcessCreateInstance
PluginManager::GetProcessCreateCallbackForPluginName(llvm::StringRef name) {
  return GetProcessInstances().GetCallbackForName(name);
}

void PluginManager::AutoCompleteProcessName(llvm::StringRef name,
                                            CompletionRequest &request) {
  for (const auto &instance : GetProcessInstances().GetSnapshot()) {
    if (instance.name.starts_with(name))
      request.AddCompletion(instance.name, instance.description);
  }
}

#pragma mark ProtocolServer

typedef PluginInstance<ProtocolServerCreateInstance> ProtocolServerInstance;
typedef PluginInstances<ProtocolServerInstance> ProtocolServerInstances;

static ProtocolServerInstances &GetProtocolServerInstances() {
  static ProtocolServerInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    ProtocolServerCreateInstance create_callback) {
  return GetProtocolServerInstances().RegisterPlugin(name, description,
                                                     create_callback);
}

bool PluginManager::UnregisterPlugin(
    ProtocolServerCreateInstance create_callback) {
  return GetProtocolServerInstances().UnregisterPlugin(create_callback);
}

llvm::StringRef
PluginManager::GetProtocolServerPluginNameAtIndex(uint32_t idx) {
  return GetProtocolServerInstances().GetNameAtIndex(idx);
}

ProtocolServerCreateInstance
PluginManager::GetProtocolCreateCallbackForPluginName(llvm::StringRef name) {
  return GetProtocolServerInstances().GetCallbackForName(name);
}

#pragma mark RegisterTypeBuilder

struct RegisterTypeBuilderInstance
    : public PluginInstance<RegisterTypeBuilderCreateInstance> {
  RegisterTypeBuilderInstance(llvm::StringRef name, llvm::StringRef description,
                              CallbackType create_callback)
      : PluginInstance<RegisterTypeBuilderCreateInstance>(name, description,
                                                          create_callback) {}
};

typedef PluginInstances<RegisterTypeBuilderInstance>
    RegisterTypeBuilderInstances;

static RegisterTypeBuilderInstances &GetRegisterTypeBuilderInstances() {
  static RegisterTypeBuilderInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    RegisterTypeBuilderCreateInstance create_callback) {
  return GetRegisterTypeBuilderInstances().RegisterPlugin(name, description,
                                                          create_callback);
}

bool PluginManager::UnregisterPlugin(
    RegisterTypeBuilderCreateInstance create_callback) {
  return GetRegisterTypeBuilderInstances().UnregisterPlugin(create_callback);
}

lldb::RegisterTypeBuilderSP
PluginManager::GetRegisterTypeBuilder(Target &target) {
  // We assume that RegisterTypeBuilderClang is the only instance of this plugin
  // type and is always present.
  auto instance = GetRegisterTypeBuilderInstances().GetInstanceAtIndex(0);
  assert(instance);
  return instance->create_callback(target);
}

#pragma mark ScriptInterpreter

struct ScriptInterpreterInstance
    : public PluginInstance<ScriptInterpreterCreateInstance> {
  ScriptInterpreterInstance(llvm::StringRef name, llvm::StringRef description,
                            CallbackType create_callback,
                            lldb::ScriptLanguage language)
      : PluginInstance<ScriptInterpreterCreateInstance>(name, description,
                                                        create_callback),
        language(language) {}

  lldb::ScriptLanguage language = lldb::eScriptLanguageNone;
};

typedef PluginInstances<ScriptInterpreterInstance> ScriptInterpreterInstances;

static ScriptInterpreterInstances &GetScriptInterpreterInstances() {
  static ScriptInterpreterInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    lldb::ScriptLanguage script_language,
    ScriptInterpreterCreateInstance create_callback) {
  return GetScriptInterpreterInstances().RegisterPlugin(
      name, description, create_callback, script_language);
}

bool PluginManager::UnregisterPlugin(
    ScriptInterpreterCreateInstance create_callback) {
  return GetScriptInterpreterInstances().UnregisterPlugin(create_callback);
}

ScriptInterpreterCreateInstance
PluginManager::GetScriptInterpreterCreateCallbackAtIndex(uint32_t idx) {
  return GetScriptInterpreterInstances().GetCallbackAtIndex(idx);
}

lldb::ScriptInterpreterSP
PluginManager::GetScriptInterpreterForLanguage(lldb::ScriptLanguage script_lang,
                                               Debugger &debugger) {
  const auto instances = GetScriptInterpreterInstances().GetSnapshot();
  ScriptInterpreterCreateInstance none_instance = nullptr;
  for (const auto &instance : instances) {
    if (instance.language == lldb::eScriptLanguageNone)
      none_instance = instance.create_callback;

    if (script_lang == instance.language)
      return instance.create_callback(debugger);
  }

  // If we didn't find one, return the ScriptInterpreter for the null language.
  assert(none_instance != nullptr);
  return none_instance(debugger);
}

#pragma mark StructuredDataPlugin

struct StructuredDataPluginInstance
    : public PluginInstance<StructuredDataPluginCreateInstance> {
  StructuredDataPluginInstance(
      llvm::StringRef name, llvm::StringRef description,
      CallbackType create_callback,
      DebuggerInitializeCallback debugger_init_callback,
      StructuredDataFilterLaunchInfo filter_callback)
      : PluginInstance<StructuredDataPluginCreateInstance>(
            name, description, create_callback, debugger_init_callback),
        filter_callback(filter_callback) {}

  StructuredDataFilterLaunchInfo filter_callback = nullptr;
};

typedef PluginInstances<StructuredDataPluginInstance>
    StructuredDataPluginInstances;

static StructuredDataPluginInstances &GetStructuredDataPluginInstances() {
  static StructuredDataPluginInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    StructuredDataPluginCreateInstance create_callback,
    DebuggerInitializeCallback debugger_init_callback,
    StructuredDataFilterLaunchInfo filter_callback) {
  return GetStructuredDataPluginInstances().RegisterPlugin(
      name, description, create_callback, debugger_init_callback,
      filter_callback);
}

bool PluginManager::UnregisterPlugin(
    StructuredDataPluginCreateInstance create_callback) {
  return GetStructuredDataPluginInstances().UnregisterPlugin(create_callback);
}

StructuredDataPluginCreateInstance
PluginManager::GetStructuredDataPluginCreateCallbackAtIndex(uint32_t idx) {
  return GetStructuredDataPluginInstances().GetCallbackAtIndex(idx);
}

StructuredDataFilterLaunchInfo
PluginManager::GetStructuredDataFilterCallbackAtIndex(
    uint32_t idx, bool &iteration_complete) {
  if (auto instance =
          GetStructuredDataPluginInstances().GetInstanceAtIndex(idx)) {
    iteration_complete = false;
    return instance->filter_callback;
  }
  iteration_complete = true;
  return nullptr;
}

#pragma mark SymbolFile

typedef PluginInstance<SymbolFileCreateInstance> SymbolFileInstance;
typedef PluginInstances<SymbolFileInstance> SymbolFileInstances;

static SymbolFileInstances &GetSymbolFileInstances() {
  static SymbolFileInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    SymbolFileCreateInstance create_callback,
    DebuggerInitializeCallback debugger_init_callback) {
  return GetSymbolFileInstances().RegisterPlugin(
      name, description, create_callback, debugger_init_callback);
}

bool PluginManager::UnregisterPlugin(SymbolFileCreateInstance create_callback) {
  return GetSymbolFileInstances().UnregisterPlugin(create_callback);
}

SymbolFileCreateInstance
PluginManager::GetSymbolFileCreateCallbackAtIndex(uint32_t idx) {
  return GetSymbolFileInstances().GetCallbackAtIndex(idx);
}

#pragma mark SymbolVendor

typedef PluginInstance<SymbolVendorCreateInstance> SymbolVendorInstance;
typedef PluginInstances<SymbolVendorInstance> SymbolVendorInstances;

static SymbolVendorInstances &GetSymbolVendorInstances() {
  static SymbolVendorInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(llvm::StringRef name,
                                   llvm::StringRef description,
                                   SymbolVendorCreateInstance create_callback) {
  return GetSymbolVendorInstances().RegisterPlugin(name, description,
                                                   create_callback);
}

bool PluginManager::UnregisterPlugin(
    SymbolVendorCreateInstance create_callback) {
  return GetSymbolVendorInstances().UnregisterPlugin(create_callback);
}

SymbolVendorCreateInstance
PluginManager::GetSymbolVendorCreateCallbackAtIndex(uint32_t idx) {
  return GetSymbolVendorInstances().GetCallbackAtIndex(idx);
}

#pragma mark SymbolLocator

struct SymbolLocatorInstance
    : public PluginInstance<SymbolLocatorCreateInstance> {
  SymbolLocatorInstance(
      llvm::StringRef name, llvm::StringRef description,
      CallbackType create_callback,
      SymbolLocatorLocateExecutableObjectFile locate_executable_object_file,
      SymbolLocatorLocateExecutableSymbolFile locate_executable_symbol_file,
      SymbolLocatorDownloadObjectAndSymbolFile download_object_symbol_file,
      SymbolLocatorFindSymbolFileInBundle find_symbol_file_in_bundle,
      DebuggerInitializeCallback debugger_init_callback)
      : PluginInstance<SymbolLocatorCreateInstance>(
            name, description, create_callback, debugger_init_callback),
        locate_executable_object_file(locate_executable_object_file),
        locate_executable_symbol_file(locate_executable_symbol_file),
        download_object_symbol_file(download_object_symbol_file),
        find_symbol_file_in_bundle(find_symbol_file_in_bundle) {}

  SymbolLocatorLocateExecutableObjectFile locate_executable_object_file;
  SymbolLocatorLocateExecutableSymbolFile locate_executable_symbol_file;
  SymbolLocatorDownloadObjectAndSymbolFile download_object_symbol_file;
  SymbolLocatorFindSymbolFileInBundle find_symbol_file_in_bundle;
};
typedef PluginInstances<SymbolLocatorInstance> SymbolLocatorInstances;

static SymbolLocatorInstances &GetSymbolLocatorInstances() {
  static SymbolLocatorInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    SymbolLocatorCreateInstance create_callback,
    SymbolLocatorLocateExecutableObjectFile locate_executable_object_file,
    SymbolLocatorLocateExecutableSymbolFile locate_executable_symbol_file,
    SymbolLocatorDownloadObjectAndSymbolFile download_object_symbol_file,
    SymbolLocatorFindSymbolFileInBundle find_symbol_file_in_bundle,
    DebuggerInitializeCallback debugger_init_callback) {
  return GetSymbolLocatorInstances().RegisterPlugin(
      name, description, create_callback, locate_executable_object_file,
      locate_executable_symbol_file, download_object_symbol_file,
      find_symbol_file_in_bundle, debugger_init_callback);
}

bool PluginManager::UnregisterPlugin(
    SymbolLocatorCreateInstance create_callback) {
  return GetSymbolLocatorInstances().UnregisterPlugin(create_callback);
}

SymbolLocatorCreateInstance
PluginManager::GetSymbolLocatorCreateCallbackAtIndex(uint32_t idx) {
  return GetSymbolLocatorInstances().GetCallbackAtIndex(idx);
}

ModuleSpec
PluginManager::LocateExecutableObjectFile(const ModuleSpec &module_spec,
                                          StatisticsMap &map) {
  auto instances = GetSymbolLocatorInstances().GetSnapshot();
  for (auto &instance : instances) {
    if (instance.locate_executable_object_file) {
      StatsDuration time;
      std::optional<ModuleSpec> result;
      {
        ElapsedTime elapsed(time);
        result = instance.locate_executable_object_file(module_spec);
      }
      map.add(instance.name, time.get().count());
      if (result)
        return *result;
    }
  }
  return {};
}

FileSpec PluginManager::LocateExecutableSymbolFile(
    const ModuleSpec &module_spec, const FileSpecList &default_search_paths,
    StatisticsMap &map) {
  auto instances = GetSymbolLocatorInstances().GetSnapshot();
  for (auto &instance : instances) {
    if (instance.locate_executable_symbol_file) {
      StatsDuration time;
      std::optional<FileSpec> result;
      {
        ElapsedTime elapsed(time);
        result = instance.locate_executable_symbol_file(module_spec,
                                                        default_search_paths);
      }
      map.add(instance.name, time.get().count());
      if (result)
        return *result;
    }
  }
  return {};
}

bool PluginManager::DownloadObjectAndSymbolFile(ModuleSpec &module_spec,
                                                Status &error,
                                                bool force_lookup,
                                                bool copy_executable) {
  auto instances = GetSymbolLocatorInstances().GetSnapshot();
  for (auto &instance : instances) {
    if (instance.download_object_symbol_file) {
      if (instance.download_object_symbol_file(module_spec, error, force_lookup,
                                               copy_executable))
        return true;
    }
  }
  return false;
}

FileSpec PluginManager::FindSymbolFileInBundle(const FileSpec &symfile_bundle,
                                               const UUID *uuid,
                                               const ArchSpec *arch) {
  auto instances = GetSymbolLocatorInstances().GetSnapshot();
  for (auto &instance : instances) {
    if (instance.find_symbol_file_in_bundle) {
      std::optional<FileSpec> result =
          instance.find_symbol_file_in_bundle(symfile_bundle, uuid, arch);
      if (result)
        return *result;
    }
  }
  return {};
}

#pragma mark Trace

struct TraceInstance
    : public PluginInstance<TraceCreateInstanceFromBundle> {
  TraceInstance(
      llvm::StringRef name, llvm::StringRef description,
      CallbackType create_callback_from_bundle,
      TraceCreateInstanceForLiveProcess create_callback_for_live_process,
      llvm::StringRef schema, DebuggerInitializeCallback debugger_init_callback)
      : PluginInstance<TraceCreateInstanceFromBundle>(
            name, description, create_callback_from_bundle,
            debugger_init_callback),
        schema(schema),
        create_callback_for_live_process(create_callback_for_live_process) {}

  llvm::StringRef schema;
  TraceCreateInstanceForLiveProcess create_callback_for_live_process;
};

typedef PluginInstances<TraceInstance> TraceInstances;

static TraceInstances &GetTracePluginInstances() {
  static TraceInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    TraceCreateInstanceFromBundle create_callback_from_bundle,
    TraceCreateInstanceForLiveProcess create_callback_for_live_process,
    llvm::StringRef schema, DebuggerInitializeCallback debugger_init_callback) {
  return GetTracePluginInstances().RegisterPlugin(
      name, description, create_callback_from_bundle,
      create_callback_for_live_process, schema, debugger_init_callback);
}

bool PluginManager::UnregisterPlugin(
    TraceCreateInstanceFromBundle create_callback_from_bundle) {
  return GetTracePluginInstances().UnregisterPlugin(
      create_callback_from_bundle);
}

TraceCreateInstanceFromBundle
PluginManager::GetTraceCreateCallback(llvm::StringRef plugin_name) {
  return GetTracePluginInstances().GetCallbackForName(plugin_name);
}

TraceCreateInstanceForLiveProcess
PluginManager::GetTraceCreateCallbackForLiveProcess(llvm::StringRef plugin_name) {
  if (auto instance = GetTracePluginInstances().GetInstanceForName(plugin_name))
    return instance->create_callback_for_live_process;

  return nullptr;
}

llvm::StringRef PluginManager::GetTraceSchema(llvm::StringRef plugin_name) {
  if (auto instance = GetTracePluginInstances().GetInstanceForName(plugin_name))
    return instance->schema;
  return llvm::StringRef();
}

llvm::StringRef PluginManager::GetTraceSchema(size_t index) {
  if (const TraceInstance *instance =
          GetTracePluginInstances().GetInstanceAtIndex(index))
    return instance->schema;
  return llvm::StringRef();
}

#pragma mark TraceExporter

struct TraceExporterInstance
    : public PluginInstance<TraceExporterCreateInstance> {
  TraceExporterInstance(
      llvm::StringRef name, llvm::StringRef description,
      TraceExporterCreateInstance create_instance,
      ThreadTraceExportCommandCreator create_thread_trace_export_command)
      : PluginInstance<TraceExporterCreateInstance>(name, description,
                                                    create_instance),
        create_thread_trace_export_command(create_thread_trace_export_command) {
  }

  ThreadTraceExportCommandCreator create_thread_trace_export_command;
};

typedef PluginInstances<TraceExporterInstance> TraceExporterInstances;

static TraceExporterInstances &GetTraceExporterInstances() {
  static TraceExporterInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    TraceExporterCreateInstance create_callback,
    ThreadTraceExportCommandCreator create_thread_trace_export_command) {
  return GetTraceExporterInstances().RegisterPlugin(
      name, description, create_callback, create_thread_trace_export_command);
}

TraceExporterCreateInstance
PluginManager::GetTraceExporterCreateCallback(llvm::StringRef plugin_name) {
  return GetTraceExporterInstances().GetCallbackForName(plugin_name);
}

bool PluginManager::UnregisterPlugin(
    TraceExporterCreateInstance create_callback) {
  return GetTraceExporterInstances().UnregisterPlugin(create_callback);
}

ThreadTraceExportCommandCreator
PluginManager::GetThreadTraceExportCommandCreatorAtIndex(uint32_t index) {
  if (const TraceExporterInstance *instance =
          GetTraceExporterInstances().GetInstanceAtIndex(index))
    return instance->create_thread_trace_export_command;
  return nullptr;
}

llvm::StringRef
PluginManager::GetTraceExporterPluginNameAtIndex(uint32_t index) {
  return GetTraceExporterInstances().GetNameAtIndex(index);
}

#pragma mark UnwindAssembly

typedef PluginInstance<UnwindAssemblyCreateInstance> UnwindAssemblyInstance;
typedef PluginInstances<UnwindAssemblyInstance> UnwindAssemblyInstances;

static UnwindAssemblyInstances &GetUnwindAssemblyInstances() {
  static UnwindAssemblyInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    UnwindAssemblyCreateInstance create_callback) {
  return GetUnwindAssemblyInstances().RegisterPlugin(name, description,
                                                     create_callback);
}

bool PluginManager::UnregisterPlugin(
    UnwindAssemblyCreateInstance create_callback) {
  return GetUnwindAssemblyInstances().UnregisterPlugin(create_callback);
}

UnwindAssemblyCreateInstance
PluginManager::GetUnwindAssemblyCreateCallbackAtIndex(uint32_t idx) {
  return GetUnwindAssemblyInstances().GetCallbackAtIndex(idx);
}

#pragma mark MemoryHistory

typedef PluginInstance<MemoryHistoryCreateInstance> MemoryHistoryInstance;
typedef PluginInstances<MemoryHistoryInstance> MemoryHistoryInstances;

static MemoryHistoryInstances &GetMemoryHistoryInstances() {
  static MemoryHistoryInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    MemoryHistoryCreateInstance create_callback) {
  return GetMemoryHistoryInstances().RegisterPlugin(name, description,
                                                    create_callback);
}

bool PluginManager::UnregisterPlugin(
    MemoryHistoryCreateInstance create_callback) {
  return GetMemoryHistoryInstances().UnregisterPlugin(create_callback);
}

MemoryHistoryCreateInstance
PluginManager::GetMemoryHistoryCreateCallbackAtIndex(uint32_t idx) {
  return GetMemoryHistoryInstances().GetCallbackAtIndex(idx);
}

#pragma mark InstrumentationRuntime

struct InstrumentationRuntimeInstance
    : public PluginInstance<InstrumentationRuntimeCreateInstance> {
  InstrumentationRuntimeInstance(
      llvm::StringRef name, llvm::StringRef description,
      CallbackType create_callback,
      InstrumentationRuntimeGetType get_type_callback)
      : PluginInstance<InstrumentationRuntimeCreateInstance>(name, description,
                                                             create_callback),
        get_type_callback(get_type_callback) {}

  InstrumentationRuntimeGetType get_type_callback = nullptr;
};

typedef PluginInstances<InstrumentationRuntimeInstance>
    InstrumentationRuntimeInstances;

static InstrumentationRuntimeInstances &GetInstrumentationRuntimeInstances() {
  static InstrumentationRuntimeInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    InstrumentationRuntimeCreateInstance create_callback,
    InstrumentationRuntimeGetType get_type_callback) {
  return GetInstrumentationRuntimeInstances().RegisterPlugin(
      name, description, create_callback, get_type_callback);
}

bool PluginManager::UnregisterPlugin(
    InstrumentationRuntimeCreateInstance create_callback) {
  return GetInstrumentationRuntimeInstances().UnregisterPlugin(create_callback);
}

InstrumentationRuntimeGetType
PluginManager::GetInstrumentationRuntimeGetTypeCallbackAtIndex(uint32_t idx) {
  if (auto instance =
          GetInstrumentationRuntimeInstances().GetInstanceAtIndex(idx))
    return instance->get_type_callback;
  return nullptr;
}

InstrumentationRuntimeCreateInstance
PluginManager::GetInstrumentationRuntimeCreateCallbackAtIndex(uint32_t idx) {
  return GetInstrumentationRuntimeInstances().GetCallbackAtIndex(idx);
}

#pragma mark TypeSystem

struct TypeSystemInstance : public PluginInstance<TypeSystemCreateInstance> {
  TypeSystemInstance(llvm::StringRef name, llvm::StringRef description,
                     CallbackType create_callback,
                     LanguageSet supported_languages_for_types,
                     LanguageSet supported_languages_for_expressions)
      : PluginInstance<TypeSystemCreateInstance>(name, description,
                                                 create_callback),
        supported_languages_for_types(supported_languages_for_types),
        supported_languages_for_expressions(
            supported_languages_for_expressions) {}

  LanguageSet supported_languages_for_types;
  LanguageSet supported_languages_for_expressions;
};

typedef PluginInstances<TypeSystemInstance> TypeSystemInstances;

static TypeSystemInstances &GetTypeSystemInstances() {
  static TypeSystemInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    TypeSystemCreateInstance create_callback,
    LanguageSet supported_languages_for_types,
    LanguageSet supported_languages_for_expressions) {
  return GetTypeSystemInstances().RegisterPlugin(
      name, description, create_callback, supported_languages_for_types,
      supported_languages_for_expressions);
}

bool PluginManager::UnregisterPlugin(TypeSystemCreateInstance create_callback) {
  return GetTypeSystemInstances().UnregisterPlugin(create_callback);
}

TypeSystemCreateInstance
PluginManager::GetTypeSystemCreateCallbackAtIndex(uint32_t idx) {
  return GetTypeSystemInstances().GetCallbackAtIndex(idx);
}

LanguageSet PluginManager::GetAllTypeSystemSupportedLanguagesForTypes() {
  const auto instances = GetTypeSystemInstances().GetSnapshot();
  LanguageSet all;
  for (unsigned i = 0; i < instances.size(); ++i)
    all.bitvector |= instances[i].supported_languages_for_types.bitvector;
  return all;
}

LanguageSet PluginManager::GetAllTypeSystemSupportedLanguagesForExpressions() {
  const auto instances = GetTypeSystemInstances().GetSnapshot();
  LanguageSet all;
  for (unsigned i = 0; i < instances.size(); ++i)
    all.bitvector |= instances[i].supported_languages_for_expressions.bitvector;
  return all;
}

#pragma mark ScriptedInterfaces

struct ScriptedInterfaceInstance
    : public PluginInstance<ScriptedInterfaceCreateInstance> {
  ScriptedInterfaceInstance(llvm::StringRef name, llvm::StringRef description,
                            ScriptedInterfaceCreateInstance create_callback,
                            lldb::ScriptLanguage language,
                            ScriptedInterfaceUsages usages)
      : PluginInstance<ScriptedInterfaceCreateInstance>(name, description,
                                                        create_callback),
        language(language), usages(usages) {}

  lldb::ScriptLanguage language;
  ScriptedInterfaceUsages usages;
};

typedef PluginInstances<ScriptedInterfaceInstance> ScriptedInterfaceInstances;

static ScriptedInterfaceInstances &GetScriptedInterfaceInstances() {
  static ScriptedInterfaceInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(
    llvm::StringRef name, llvm::StringRef description,
    ScriptedInterfaceCreateInstance create_callback,
    lldb::ScriptLanguage language, ScriptedInterfaceUsages usages) {
  return GetScriptedInterfaceInstances().RegisterPlugin(
      name, description, create_callback, language, usages);
}

bool PluginManager::UnregisterPlugin(
    ScriptedInterfaceCreateInstance create_callback) {
  return GetScriptedInterfaceInstances().UnregisterPlugin(create_callback);
}

uint32_t PluginManager::GetNumScriptedInterfaces() {
  return GetScriptedInterfaceInstances().GetSnapshot().size();
}

llvm::StringRef PluginManager::GetScriptedInterfaceNameAtIndex(uint32_t index) {
  return GetScriptedInterfaceInstances().GetNameAtIndex(index);
}

llvm::StringRef
PluginManager::GetScriptedInterfaceDescriptionAtIndex(uint32_t index) {
  return GetScriptedInterfaceInstances().GetDescriptionAtIndex(index);
}

lldb::ScriptLanguage
PluginManager::GetScriptedInterfaceLanguageAtIndex(uint32_t idx) {
  if (auto instance = GetScriptedInterfaceInstances().GetInstanceAtIndex(idx))
    return instance->language;
  return ScriptLanguage::eScriptLanguageNone;
}

ScriptedInterfaceUsages
PluginManager::GetScriptedInterfaceUsagesAtIndex(uint32_t idx) {
  if (auto instance = GetScriptedInterfaceInstances().GetInstanceAtIndex(idx))
    return instance->usages;
  return {};
}

#pragma mark REPL

struct REPLInstance : public PluginInstance<REPLCreateInstance> {
  REPLInstance(llvm::StringRef name, llvm::StringRef description,
               CallbackType create_callback, LanguageSet supported_languages)
      : PluginInstance<REPLCreateInstance>(name, description, create_callback),
        supported_languages(supported_languages) {}

  LanguageSet supported_languages;
};

typedef PluginInstances<REPLInstance> REPLInstances;

static REPLInstances &GetREPLInstances() {
  static REPLInstances g_instances;
  return g_instances;
}

bool PluginManager::RegisterPlugin(llvm::StringRef name, llvm::StringRef description,
                                   REPLCreateInstance create_callback,
                                   LanguageSet supported_languages) {
  return GetREPLInstances().RegisterPlugin(name, description, create_callback,
                                           supported_languages);
}

bool PluginManager::UnregisterPlugin(REPLCreateInstance create_callback) {
  return GetREPLInstances().UnregisterPlugin(create_callback);
}

REPLCreateInstance PluginManager::GetREPLCreateCallbackAtIndex(uint32_t idx) {
  return GetREPLInstances().GetCallbackAtIndex(idx);
}

LanguageSet PluginManager::GetREPLSupportedLanguagesAtIndex(uint32_t idx) {
  if (auto instance = GetREPLInstances().GetInstanceAtIndex(idx))
    return instance->supported_languages;
  return LanguageSet();
}

LanguageSet PluginManager::GetREPLAllTypeSystemSupportedLanguages() {
  const auto instances = GetREPLInstances().GetSnapshot();
  LanguageSet all;
  for (unsigned i = 0; i < instances.size(); ++i)
    all.bitvector |= instances[i].supported_languages.bitvector;
  return all;
}

#pragma mark PluginManager

void PluginManager::DebuggerInitialize(Debugger &debugger) {
  GetDynamicLoaderInstances().PerformDebuggerCallback(debugger);
  GetJITLoaderInstances().PerformDebuggerCallback(debugger);
  GetObjectFileInstances().PerformDebuggerCallback(debugger);
  GetPlatformInstances().PerformDebuggerCallback(debugger);
  GetProcessInstances().PerformDebuggerCallback(debugger);
  GetSymbolFileInstances().PerformDebuggerCallback(debugger);
  GetSymbolLocatorInstances().PerformDebuggerCallback(debugger);
  GetOperatingSystemInstances().PerformDebuggerCallback(debugger);
  GetStructuredDataPluginInstances().PerformDebuggerCallback(debugger);
  GetTracePluginInstances().PerformDebuggerCallback(debugger);
  GetScriptedInterfaceInstances().PerformDebuggerCallback(debugger);
  GetLanguageInstances().PerformDebuggerCallback(debugger);
}

// This is the preferred new way to register plugin specific settings.  e.g.
// This will put a plugin's settings under e.g.
// "plugin.<plugin_type_name>.<plugin_type_desc>.SETTINGNAME".
static lldb::OptionValuePropertiesSP
GetDebuggerPropertyForPlugins(Debugger &debugger, llvm::StringRef plugin_type_name,
                              llvm::StringRef plugin_type_desc,
                              bool can_create) {
  lldb::OptionValuePropertiesSP parent_properties_sp(
      debugger.GetValueProperties());
  if (parent_properties_sp) {
    static constexpr llvm::StringLiteral g_property_name("plugin");

    OptionValuePropertiesSP plugin_properties_sp =
        parent_properties_sp->GetSubProperty(nullptr, g_property_name);
    if (!plugin_properties_sp && can_create) {
      plugin_properties_sp =
          std::make_shared<OptionValueProperties>(g_property_name);
      parent_properties_sp->AppendProperty(g_property_name,
                                           "Settings specify to plugins.", true,
                                           plugin_properties_sp);
    }

    if (plugin_properties_sp) {
      lldb::OptionValuePropertiesSP plugin_type_properties_sp =
          plugin_properties_sp->GetSubProperty(nullptr, plugin_type_name);
      if (!plugin_type_properties_sp && can_create) {
        plugin_type_properties_sp =
            std::make_shared<OptionValueProperties>(plugin_type_name);
        plugin_properties_sp->AppendProperty(plugin_type_name, plugin_type_desc,
                                             true, plugin_type_properties_sp);
      }
      return plugin_type_properties_sp;
    }
  }
  return lldb::OptionValuePropertiesSP();
}

// This is deprecated way to register plugin specific settings.  e.g.
// "<plugin_type_name>.plugin.<plugin_type_desc>.SETTINGNAME" and Platform
// generic settings would be under "platform.SETTINGNAME".
static lldb::OptionValuePropertiesSP GetDebuggerPropertyForPluginsOldStyle(
    Debugger &debugger, llvm::StringRef plugin_type_name,
    llvm::StringRef plugin_type_desc, bool can_create) {
  static constexpr llvm::StringLiteral g_property_name("plugin");
  lldb::OptionValuePropertiesSP parent_properties_sp(
      debugger.GetValueProperties());
  if (parent_properties_sp) {
    OptionValuePropertiesSP plugin_properties_sp =
        parent_properties_sp->GetSubProperty(nullptr, plugin_type_name);
    if (!plugin_properties_sp && can_create) {
      plugin_properties_sp =
          std::make_shared<OptionValueProperties>(plugin_type_name);
      parent_properties_sp->AppendProperty(plugin_type_name, plugin_type_desc,
                                           true, plugin_properties_sp);
    }

    if (plugin_properties_sp) {
      lldb::OptionValuePropertiesSP plugin_type_properties_sp =
          plugin_properties_sp->GetSubProperty(nullptr, g_property_name);
      if (!plugin_type_properties_sp && can_create) {
        plugin_type_properties_sp =
            std::make_shared<OptionValueProperties>(g_property_name);
        plugin_properties_sp->AppendProperty(g_property_name,
                                             "Settings specific to plugins",
                                             true, plugin_type_properties_sp);
      }
      return plugin_type_properties_sp;
    }
  }
  return lldb::OptionValuePropertiesSP();
}

namespace {

typedef lldb::OptionValuePropertiesSP
GetDebuggerPropertyForPluginsPtr(Debugger &, llvm::StringRef, llvm::StringRef,
                                 bool can_create);
}

static lldb::OptionValuePropertiesSP
GetSettingForPlugin(Debugger &debugger, llvm::StringRef setting_name,
                    llvm::StringRef plugin_type_name,
                    GetDebuggerPropertyForPluginsPtr get_debugger_property =
                        GetDebuggerPropertyForPlugins) {
  lldb::OptionValuePropertiesSP properties_sp;
  lldb::OptionValuePropertiesSP plugin_type_properties_sp(get_debugger_property(
      debugger, plugin_type_name,
      "", // not creating to so we don't need the description
      false));
  if (plugin_type_properties_sp)
    properties_sp =
        plugin_type_properties_sp->GetSubProperty(nullptr, setting_name);
  return properties_sp;
}

static bool
CreateSettingForPlugin(Debugger &debugger, llvm::StringRef plugin_type_name,
                       llvm::StringRef plugin_type_desc,
                       const lldb::OptionValuePropertiesSP &properties_sp,
                       llvm::StringRef description, bool is_global_property,
                       GetDebuggerPropertyForPluginsPtr get_debugger_property =
                           GetDebuggerPropertyForPlugins) {
  if (properties_sp) {
    lldb::OptionValuePropertiesSP plugin_type_properties_sp(
        get_debugger_property(debugger, plugin_type_name, plugin_type_desc,
                              true));
    if (plugin_type_properties_sp) {
      plugin_type_properties_sp->AppendProperty(properties_sp->GetName(),
                                                description, is_global_property,
                                                properties_sp);
      return true;
    }
  }
  return false;
}

static constexpr llvm::StringLiteral kDynamicLoaderPluginName("dynamic-loader");
static constexpr llvm::StringLiteral kPlatformPluginName("platform");
static constexpr llvm::StringLiteral kProcessPluginName("process");
static constexpr llvm::StringLiteral kTracePluginName("trace");
static constexpr llvm::StringLiteral kObjectFilePluginName("object-file");
static constexpr llvm::StringLiteral kSymbolFilePluginName("symbol-file");
static constexpr llvm::StringLiteral kSymbolLocatorPluginName("symbol-locator");
static constexpr llvm::StringLiteral kJITLoaderPluginName("jit-loader");
static constexpr llvm::StringLiteral
    kStructuredDataPluginName("structured-data");
static constexpr llvm::StringLiteral kCPlusPlusLanguagePlugin("cplusplus");

lldb::OptionValuePropertiesSP
PluginManager::GetSettingForDynamicLoaderPlugin(Debugger &debugger,
                                                llvm::StringRef setting_name) {
  return GetSettingForPlugin(debugger, setting_name, kDynamicLoaderPluginName);
}

bool PluginManager::CreateSettingForDynamicLoaderPlugin(
    Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  return CreateSettingForPlugin(debugger, kDynamicLoaderPluginName,
                                "Settings for dynamic loader plug-ins",
                                properties_sp, description, is_global_property);
}

lldb::OptionValuePropertiesSP
PluginManager::GetSettingForPlatformPlugin(Debugger &debugger,
                                           llvm::StringRef setting_name) {
  return GetSettingForPlugin(debugger, setting_name, kPlatformPluginName,
                             GetDebuggerPropertyForPluginsOldStyle);
}

bool PluginManager::CreateSettingForPlatformPlugin(
    Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  return CreateSettingForPlugin(debugger, kPlatformPluginName,
                                "Settings for platform plug-ins", properties_sp,
                                description, is_global_property,
                                GetDebuggerPropertyForPluginsOldStyle);
}

lldb::OptionValuePropertiesSP
PluginManager::GetSettingForProcessPlugin(Debugger &debugger,
                                          llvm::StringRef setting_name) {
  return GetSettingForPlugin(debugger, setting_name, kProcessPluginName);
}

bool PluginManager::CreateSettingForProcessPlugin(
    Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  return CreateSettingForPlugin(debugger, kProcessPluginName,
                                "Settings for process plug-ins", properties_sp,
                                description, is_global_property);
}

lldb::OptionValuePropertiesSP
PluginManager::GetSettingForSymbolLocatorPlugin(Debugger &debugger,
                                                llvm::StringRef setting_name) {
  return GetSettingForPlugin(debugger, setting_name, kSymbolLocatorPluginName);
}

bool PluginManager::CreateSettingForSymbolLocatorPlugin(
    Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  return CreateSettingForPlugin(debugger, kSymbolLocatorPluginName,
                                "Settings for symbol locator plug-ins",
                                properties_sp, description, is_global_property);
}

bool PluginManager::CreateSettingForTracePlugin(
    Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  return CreateSettingForPlugin(debugger, kTracePluginName,
                                "Settings for trace plug-ins", properties_sp,
                                description, is_global_property);
}

lldb::OptionValuePropertiesSP
PluginManager::GetSettingForObjectFilePlugin(Debugger &debugger,
                                             llvm::StringRef setting_name) {
  return GetSettingForPlugin(debugger, setting_name, kObjectFilePluginName);
}

bool PluginManager::CreateSettingForObjectFilePlugin(
    Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  return CreateSettingForPlugin(debugger, kObjectFilePluginName,
                                "Settings for object file plug-ins",
                                properties_sp, description, is_global_property);
}

lldb::OptionValuePropertiesSP
PluginManager::GetSettingForSymbolFilePlugin(Debugger &debugger,
                                             llvm::StringRef setting_name) {
  return GetSettingForPlugin(debugger, setting_name, kSymbolFilePluginName);
}

bool PluginManager::CreateSettingForSymbolFilePlugin(
    Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  return CreateSettingForPlugin(debugger, kSymbolFilePluginName,
                                "Settings for symbol file plug-ins",
                                properties_sp, description, is_global_property);
}

lldb::OptionValuePropertiesSP
PluginManager::GetSettingForJITLoaderPlugin(Debugger &debugger,
                                            llvm::StringRef setting_name) {
  return GetSettingForPlugin(debugger, setting_name, kJITLoaderPluginName);
}

bool PluginManager::CreateSettingForJITLoaderPlugin(
    Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  return CreateSettingForPlugin(debugger, kJITLoaderPluginName,
                                "Settings for JIT loader plug-ins",
                                properties_sp, description, is_global_property);
}

static const char *kOperatingSystemPluginName("os");

lldb::OptionValuePropertiesSP
PluginManager::GetSettingForOperatingSystemPlugin(Debugger &debugger,
                                                  llvm::StringRef setting_name) {
  lldb::OptionValuePropertiesSP properties_sp;
  lldb::OptionValuePropertiesSP plugin_type_properties_sp(
      GetDebuggerPropertyForPlugins(
          debugger, kOperatingSystemPluginName,
          "", // not creating to so we don't need the description
          false));
  if (plugin_type_properties_sp)
    properties_sp =
        plugin_type_properties_sp->GetSubProperty(nullptr, setting_name);
  return properties_sp;
}

bool PluginManager::CreateSettingForOperatingSystemPlugin(
    Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  if (properties_sp) {
    lldb::OptionValuePropertiesSP plugin_type_properties_sp(
        GetDebuggerPropertyForPlugins(debugger, kOperatingSystemPluginName,
                                      "Settings for operating system plug-ins",
                                      true));
    if (plugin_type_properties_sp) {
      plugin_type_properties_sp->AppendProperty(properties_sp->GetName(),
                                                description, is_global_property,
                                                properties_sp);
      return true;
    }
  }
  return false;
}

lldb::OptionValuePropertiesSP
PluginManager::GetSettingForStructuredDataPlugin(Debugger &debugger,
                                                 llvm::StringRef setting_name) {
  return GetSettingForPlugin(debugger, setting_name, kStructuredDataPluginName);
}

bool PluginManager::CreateSettingForStructuredDataPlugin(
    Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  return CreateSettingForPlugin(debugger, kStructuredDataPluginName,
                                "Settings for structured data plug-ins",
                                properties_sp, description, is_global_property);
}

lldb::OptionValuePropertiesSP
PluginManager::GetSettingForCPlusPlusLanguagePlugin(
    Debugger &debugger, llvm::StringRef setting_name) {
  return GetSettingForPlugin(debugger, setting_name, kCPlusPlusLanguagePlugin);
}

bool PluginManager::CreateSettingForCPlusPlusLanguagePlugin(
    Debugger &debugger, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  return CreateSettingForPlugin(debugger, kCPlusPlusLanguagePlugin,
                                "Settings for CPlusPlus language plug-ins",
                                properties_sp, description, is_global_property);
}

//
// Plugin Info+Enable Implementations
//
std::vector<RegisteredPluginInfo> PluginManager::GetABIPluginInfo() {
  return GetABIInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetABIPluginEnabled(llvm::StringRef name, bool enable) {
  return GetABIInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetArchitecturePluginInfo() {
  return GetArchitectureInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetArchitecturePluginEnabled(llvm::StringRef name,
                                                 bool enable) {
  return GetArchitectureInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetDisassemblerPluginInfo() {
  return GetDisassemblerInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetDisassemblerPluginEnabled(llvm::StringRef name,
                                                 bool enable) {
  return GetDisassemblerInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetDynamicLoaderPluginInfo() {
  return GetDynamicLoaderInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetDynamicLoaderPluginEnabled(llvm::StringRef name,
                                                  bool enable) {
  return GetDynamicLoaderInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo>
PluginManager::GetEmulateInstructionPluginInfo() {
  return GetEmulateInstructionInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetEmulateInstructionPluginEnabled(llvm::StringRef name,
                                                       bool enable) {
  return GetEmulateInstructionInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo>
PluginManager::GetInstrumentationRuntimePluginInfo() {
  return GetInstrumentationRuntimeInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetInstrumentationRuntimePluginEnabled(llvm::StringRef name,
                                                           bool enable) {
  return GetInstrumentationRuntimeInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetJITLoaderPluginInfo() {
  return GetJITLoaderInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetJITLoaderPluginEnabled(llvm::StringRef name,
                                              bool enable) {
  return GetJITLoaderInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetLanguagePluginInfo() {
  return GetLanguageInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetLanguagePluginEnabled(llvm::StringRef name,
                                             bool enable) {
  return GetLanguageInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo>
PluginManager::GetLanguageRuntimePluginInfo() {
  return GetLanguageRuntimeInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetLanguageRuntimePluginEnabled(llvm::StringRef name,
                                                    bool enable) {
  return GetLanguageRuntimeInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetMemoryHistoryPluginInfo() {
  return GetMemoryHistoryInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetMemoryHistoryPluginEnabled(llvm::StringRef name,
                                                  bool enable) {
  return GetMemoryHistoryInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo>
PluginManager::GetObjectContainerPluginInfo() {
  return GetObjectContainerInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetObjectContainerPluginEnabled(llvm::StringRef name,
                                                    bool enable) {
  return GetObjectContainerInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetObjectFilePluginInfo() {
  return GetObjectFileInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetObjectFilePluginEnabled(llvm::StringRef name,
                                               bool enable) {
  return GetObjectFileInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo>
PluginManager::GetOperatingSystemPluginInfo() {
  return GetOperatingSystemInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetOperatingSystemPluginEnabled(llvm::StringRef name,
                                                    bool enable) {
  return GetOperatingSystemInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetPlatformPluginInfo() {
  return GetPlatformInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetPlatformPluginEnabled(llvm::StringRef name,
                                             bool enable) {
  return GetPlatformInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetProcessPluginInfo() {
  return GetProcessInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetProcessPluginEnabled(llvm::StringRef name, bool enable) {
  return GetProcessInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetREPLPluginInfo() {
  return GetREPLInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetREPLPluginEnabled(llvm::StringRef name, bool enable) {
  return GetREPLInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo>
PluginManager::GetRegisterTypeBuilderPluginInfo() {
  return GetRegisterTypeBuilderInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetRegisterTypeBuilderPluginEnabled(llvm::StringRef name,
                                                        bool enable) {
  return GetRegisterTypeBuilderInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo>
PluginManager::GetScriptInterpreterPluginInfo() {
  return GetScriptInterpreterInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetScriptInterpreterPluginEnabled(llvm::StringRef name,
                                                      bool enable) {
  return GetScriptInterpreterInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo>
PluginManager::GetScriptedInterfacePluginInfo() {
  return GetScriptedInterfaceInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetScriptedInterfacePluginEnabled(llvm::StringRef name,
                                                      bool enable) {
  return GetScriptedInterfaceInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetStructuredDataPluginInfo() {
  return GetStructuredDataPluginInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetStructuredDataPluginEnabled(llvm::StringRef name,
                                                   bool enable) {
  return GetStructuredDataPluginInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetSymbolFilePluginInfo() {
  return GetSymbolFileInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetSymbolFilePluginEnabled(llvm::StringRef name,
                                               bool enable) {
  return GetSymbolFileInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetSymbolLocatorPluginInfo() {
  return GetSymbolLocatorInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetSymbolLocatorPluginEnabled(llvm::StringRef name,
                                                  bool enable) {
  return GetSymbolLocatorInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetSymbolVendorPluginInfo() {
  return GetSymbolVendorInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetSymbolVendorPluginEnabled(llvm::StringRef name,
                                                 bool enable) {
  return GetSymbolVendorInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetSystemRuntimePluginInfo() {
  return GetSystemRuntimeInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetSystemRuntimePluginEnabled(llvm::StringRef name,
                                                  bool enable) {
  return GetSystemRuntimeInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetTracePluginInfo() {
  return GetTracePluginInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetTracePluginEnabled(llvm::StringRef name, bool enable) {
  return GetTracePluginInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetTraceExporterPluginInfo() {
  return GetTraceExporterInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetTraceExporterPluginEnabled(llvm::StringRef name,
                                                  bool enable) {
  return GetTraceExporterInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetTypeSystemPluginInfo() {
  return GetTypeSystemInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetTypeSystemPluginEnabled(llvm::StringRef name,
                                               bool enable) {
  return GetTypeSystemInstances().SetInstanceEnabled(name, enable);
}

std::vector<RegisteredPluginInfo> PluginManager::GetUnwindAssemblyPluginInfo() {
  return GetUnwindAssemblyInstances().GetPluginInfoForAllInstances();
}
bool PluginManager::SetUnwindAssemblyPluginEnabled(llvm::StringRef name,
                                                   bool enable) {
  return GetUnwindAssemblyInstances().SetInstanceEnabled(name, enable);
}

void PluginManager::AutoCompletePluginName(llvm::StringRef name,
                                           CompletionRequest &request) {
  // Split the name into the namespace and the plugin name.
  // If there is no dot then the ns_name will be equal to name and
  // plugin_prefix will be empty.
  llvm::StringRef ns_name, plugin_prefix;
  std::tie(ns_name, plugin_prefix) = name.split('.');

  for (const PluginNamespace &plugin_ns : GetPluginNamespaces()) {
    // If the plugin namespace matches exactly then
    // add all the plugins in this namespace as completions if the
    // plugin names starts with the plugin_prefix. If the plugin_prefix
    // is empty then it will match all the plugins (empty string is a
    // prefix of everything).
    if (plugin_ns.name == ns_name) {
      for (const RegisteredPluginInfo &plugin : plugin_ns.get_info()) {
        llvm::SmallString<128> buf;
        if (plugin.name.starts_with(plugin_prefix))
          request.AddCompletion(
              (plugin_ns.name + "." + plugin.name).toStringRef(buf));
      }
    } else if (plugin_ns.name.starts_with(name) &&
               !plugin_ns.get_info().empty()) {
      // Otherwise check if the namespace is a prefix of the full name.
      // Use a partial completion here so that we can either operate on the full
      // namespace or tab-complete to the next level.
      request.AddCompletion(plugin_ns.name, "", CompletionMode::Partial);
    }
  }
}
