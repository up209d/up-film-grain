

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <label>{label}</label>
      <div className="fieldbody">{children}</div>
    </div>
  );
}

export default Field;
