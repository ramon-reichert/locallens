//go:build !windows && !darwin

package sysmon

// Capture returns an empty snapshot on non-Windows platforms.
func Capture() Snapshot {
	return Snapshot{}
}
