//go:build windows

package sysmon

import (
	"syscall"
	"unsafe"
)

var (
	kernel32              = syscall.NewLazyDLL("kernel32.dll")
	psapi                 = syscall.NewLazyDLL("psapi.dll")
	globalMemoryStatusEx  = kernel32.NewProc("GlobalMemoryStatusEx")
	getProcessMemoryInfo  = psapi.NewProc("GetProcessMemoryInfo")
	getCurrentProcess     = kernel32.NewProc("GetCurrentProcess")
)

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

type processMemoryCounters struct {
	CB                         uint32
	PageFaultCount             uint32
	PeakWorkingSetSize         uintptr
	WorkingSetSize             uintptr
	QuotaPeakPagedPoolUsage    uintptr
	QuotaPagedPoolUsage        uintptr
	QuotaPeakNonPagedPoolUsage uintptr
	QuotaNonPagedPoolUsage     uintptr
	PagefileUsage              uintptr
	PeakPagefileUsage          uintptr
}

// Snapshot holds system memory metrics at a point in time.
type Snapshot struct {
	AvailableRAM_MB uint64
	PageFaults      uint64
}

// Capture takes a snapshot of current system memory state.
func Capture() Snapshot {
	var snap Snapshot

	// Get available RAM
	var memStatus memoryStatusEx
	memStatus.Length = uint32(unsafe.Sizeof(memStatus))
	ret, _, _ := globalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&memStatus)))
	if ret != 0 {
		snap.AvailableRAM_MB = memStatus.AvailPhys / (1024 * 1024)
	}

	// Get page faults for current process
	handle, _, _ := getCurrentProcess.Call()
	var memCounters processMemoryCounters
	memCounters.CB = uint32(unsafe.Sizeof(memCounters))
	ret, _, _ = getProcessMemoryInfo.Call(
		handle,
		uintptr(unsafe.Pointer(&memCounters)),
		uintptr(memCounters.CB),
	)
	if ret != 0 {
		snap.PageFaults = uint64(memCounters.PageFaultCount)
	}

	return snap
}
