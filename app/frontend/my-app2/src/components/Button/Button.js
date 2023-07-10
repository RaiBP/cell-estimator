const Button = ({ children, onClick }) => {
  const style = {
    display: 'block',
    padding: '20px 20px',
    marginBottom: '10px',
    backgroundColor: '#5A5F6C',
    color: '#FFF',
    borderRadius: '4px',
    textAlign: 'center',
    cursor: 'pointer',
    width: '80%',
    height: '7%',
    fontSize: '100%',
  }

  return (
    <button onClick={onClick} style={style}>
      {children}
    </button>
  )
}

export default Button
