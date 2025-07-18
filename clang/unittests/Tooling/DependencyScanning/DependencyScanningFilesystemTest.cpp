//===- DependencyScanningFilesystemTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningFilesystem.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

using namespace clang::tooling::dependencies;

TEST(DependencyScanningWorkerFilesystem, CacheStatusFailures) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();

  auto InstrumentingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(InMemoryFS);

  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InstrumentingFS);
  DependencyScanningWorkerFilesystem DepFS2(SharedCache, InstrumentingFS);

  DepFS.status("/foo.c");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 1u);

  DepFS.status("/foo.c");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 1u); // Cached, no increase.

  DepFS.status("/bar.c");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);

  DepFS2.status("/foo.c");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u); // Shared cache.
}

TEST(DependencyScanningFilesystem, CacheGetRealPath) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory("/");
  InMemoryFS->addFile("/foo", 0, llvm::MemoryBuffer::getMemBuffer(""));
  InMemoryFS->addFile("/bar", 0, llvm::MemoryBuffer::getMemBuffer(""));

  auto InstrumentingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(InMemoryFS);

  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InstrumentingFS);
  DependencyScanningWorkerFilesystem DepFS2(SharedCache, InstrumentingFS);

  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/foo", Result);
    EXPECT_EQ(InstrumentingFS->NumGetRealPathCalls, 1u);
  }

  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/foo", Result);
    EXPECT_EQ(InstrumentingFS->NumGetRealPathCalls, 1u); // Cached, no increase.
  }

  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/bar", Result);
    EXPECT_EQ(InstrumentingFS->NumGetRealPathCalls, 2u);
  }

  {
    llvm::SmallString<128> Result;
    DepFS2.getRealPath("/foo", Result);
    EXPECT_EQ(InstrumentingFS->NumGetRealPathCalls, 2u); // Shared cache.
  }
}

TEST(DependencyScanningFilesystem, RealPathAndStatusInvariants) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory("/");
  InMemoryFS->addFile("/foo.c", 0, llvm::MemoryBuffer::getMemBuffer(""));
  InMemoryFS->addFile("/bar.c", 0, llvm::MemoryBuffer::getMemBuffer(""));

  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InMemoryFS);

  // Success.
  {
    DepFS.status("/foo.c");

    llvm::SmallString<128> Result;
    DepFS.getRealPath("/foo.c", Result);
  }
  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/bar.c", Result);

    DepFS.status("/bar.c");
  }

  // Failure.
  {
    DepFS.status("/foo.m");

    llvm::SmallString<128> Result;
    DepFS.getRealPath("/foo.m", Result);
  }
  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/bar.m", Result);

    DepFS.status("/bar.m");
  }

  // Failure without caching.
  {
    DepFS.status("/foo");

    llvm::SmallString<128> Result;
    DepFS.getRealPath("/foo", Result);
  }
  {
    llvm::SmallString<128> Result;
    DepFS.getRealPath("/bar", Result);

    DepFS.status("/bar");
  }
}

TEST(DependencyScanningFilesystem, CacheStatOnExists) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  auto InstrumentingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(InMemoryFS);
  InMemoryFS->setCurrentWorkingDirectory("/");
  InMemoryFS->addFile("/foo", 0, llvm::MemoryBuffer::getMemBuffer(""));
  InMemoryFS->addFile("/bar", 0, llvm::MemoryBuffer::getMemBuffer(""));
  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InstrumentingFS);

  DepFS.status("/foo");
  DepFS.status("/foo");
  DepFS.status("/bar");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);

  DepFS.exists("/foo");
  DepFS.exists("/bar");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);
  EXPECT_EQ(InstrumentingFS->NumExistsCalls, 0u);
}

TEST(DependencyScanningFilesystem, CacheStatFailures) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory("/");
  InMemoryFS->addFile("/dir/vector", 0, llvm::MemoryBuffer::getMemBuffer(""));
  InMemoryFS->addFile("/cache/a.pcm", 0, llvm::MemoryBuffer::getMemBuffer(""));

  auto InstrumentingFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::TracingFileSystem>(InMemoryFS);

  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InstrumentingFS);

  DepFS.status("/dir");
  DepFS.status("/dir");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 1u);

  DepFS.status("/dir/vector");
  DepFS.status("/dir/vector");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 2u);

  DepFS.setBypassedPathPrefix("/cache");
  DepFS.exists("/cache/a.pcm");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 3u);
  DepFS.exists("/cache/a.pcm");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 4u);

  DepFS.resetBypassedPathPrefix();
  DepFS.exists("/cache/a.pcm");
  DepFS.exists("/cache/a.pcm");
  EXPECT_EQ(InstrumentingFS->NumStatusCalls, 5u);
}

TEST(DependencyScanningFilesystem, DiagnoseStaleStatFailures) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory("/");

  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InMemoryFS);

  bool Path1Exists = DepFS.exists("/path1.suffix");
  ASSERT_EQ(Path1Exists, false);

  // Adding a file that has been stat-ed,
  InMemoryFS->addFile("/path1.suffix", 0, llvm::MemoryBuffer::getMemBuffer(""));
  Path1Exists = DepFS.exists("/path1.suffix");
  // Due to caching in SharedCache, path1 should not exist in
  // DepFS's eyes.
  ASSERT_EQ(Path1Exists, false);

  auto InvalidEntries = SharedCache.getOutOfDateEntries(*InMemoryFS);

  EXPECT_EQ(InvalidEntries.size(), 1u);
  ASSERT_STREQ("/path1.suffix", InvalidEntries[0].Path);
}

TEST(DependencyScanningFilesystem, DiagnoseCachedFileSizeChange) {
  auto InMemoryFS1 = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  auto InMemoryFS2 = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS1->setCurrentWorkingDirectory("/");
  InMemoryFS2->setCurrentWorkingDirectory("/");

  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, InMemoryFS1);

  InMemoryFS1->addFile("/path1.suffix", 0,
                       llvm::MemoryBuffer::getMemBuffer(""));
  bool Path1Exists = DepFS.exists("/path1.suffix");
  ASSERT_EQ(Path1Exists, true);

  // Add a file to a new FS that has the same path but different content.
  InMemoryFS2->addFile("/path1.suffix", 1,
                       llvm::MemoryBuffer::getMemBuffer("        "));

  // Check against the new file system. InMemoryFS2 could be the underlying
  // physical system in the real world.
  auto InvalidEntries = SharedCache.getOutOfDateEntries(*InMemoryFS2);

  ASSERT_EQ(InvalidEntries.size(), 1u);
  ASSERT_STREQ("/path1.suffix", InvalidEntries[0].Path);
  auto SizeInfo = std::get_if<
      DependencyScanningFilesystemSharedCache::OutOfDateEntry::SizeChangedInfo>(
      &InvalidEntries[0].Info);
  ASSERT_TRUE(SizeInfo);
  ASSERT_EQ(SizeInfo->CachedSize, 0u);
  ASSERT_EQ(SizeInfo->ActualSize, 8u);
}

TEST(DependencyScanningFilesystem, DoNotDiagnoseDirSizeChange) {
  llvm::SmallString<128> Dir;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("tmp", Dir));

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS =
      llvm::vfs::createPhysicalFileSystem();

  DependencyScanningFilesystemSharedCache SharedCache;
  DependencyScanningWorkerFilesystem DepFS(SharedCache, FS);

  // Trigger the file system cache.
  ASSERT_EQ(DepFS.exists(Dir), true);

  // Add a file to the FS to change its size.
  // It seems that directory sizes reported are not meaningful,
  // and should not be used to check for size changes.
  // This test is setup only to trigger a size change so that we
  // know we are excluding directories from reporting.
  llvm::SmallString<128> FilePath = Dir;
  llvm::sys::path::append(FilePath, "file.h");
  {
    std::error_code EC;
    llvm::raw_fd_ostream TempFile(FilePath, EC);
    ASSERT_FALSE(EC);
  }

  // We do not report directory size changes.
  auto InvalidEntries = SharedCache.getOutOfDateEntries(*FS);
  EXPECT_EQ(InvalidEntries.size(), 0u);
}
