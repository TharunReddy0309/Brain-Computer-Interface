import { useRef, useState } from "react";

function formatBytes(bytes) {
  return bytes >= 1024 ** 3
    ? `${(bytes / 1024 ** 3).toFixed(1)} GB`
    : `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

export default function FileDropzone({
  file,
  error,
  disabled,
  onSelect,
  title = "Labelled Dataset",
  extension = ".zip",
  description = "Drop your dataset archive here",
  format = "ZIP archive",
  icon = "dataset-archive.svg",
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  function chooseFile(event) {
    onSelect(event.target.files?.[0] ?? null);
    event.target.value = "";
  }

  function dropFile(event) {
    event.preventDefault();
    setDragging(false);
    if (!disabled) onSelect(event.dataTransfer.files?.[0] ?? null);
  }

  return (
    <div
      className={`dropzone ${dragging ? "dragging" : ""} ${error ? "has-error" : ""}`}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label={`Choose ${title} file, ${extension} only`}
      aria-disabled={disabled}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(event) => {
        if (!disabled && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={dropFile}
    >
      <input
        ref={inputRef}
        type="file"
        accept={extension}
        disabled={disabled}
        onChange={chooseFile}
        tabIndex={-1}
        aria-hidden="true"
      />
      <span className="dropzone-icon">
        <img src={`/assets/evaluate/${icon}`} alt="" />
      </span>
      <span className="dropzone-title">
        {title}
        <span className="file-badge dataset">{extension} only</span>
      </span>
      <span className="dropzone-description">
        {file ? file.name : description}
      </span>
      <span className="dropzone-format">
        {file ? formatBytes(file.size) : format}
      </span>
      <span className="browse-button">{file ? "Change file" : "Browse file"}</span>
      {error && <span className="field-error">{error}</span>}
    </div>
  );
}
