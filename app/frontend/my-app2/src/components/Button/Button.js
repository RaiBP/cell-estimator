import './Button.css'

const Button = ({ children, onClick }) => {
  return (
    <button className='menu-button' onClick={onClick}>
      {children}
    </button>
  )
}

export default Button
